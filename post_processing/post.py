from astropy.table import Table, join, MaskedColumn, vstack, Column
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import fnmatch
import h5py
import lupita
import flair.flares as flares
from scipy.integrate import trapezoid
import astropy.units as u
from scipy.optimize import curve_fit
from jim import FFD

import matplotlib.colors as mcol
import matplotlib.cm as cm

import logging
import argparse

def count_sector_files_in_output_dir(directory, tic_id):
    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, str(tic_id)+'*') and not fnmatch.fnmatch(file, '*sector*') and not fnmatch.fnmatch(file, '*final*'):
            count += 1
    return count

def access_data(tic_id, output_dir):
    """
    Access the data from the output files for a given TIC ID.
    
    Parameters:
    - tic_id: The TIC ID of the target.
    
    Returns:
    - sector_specific_data: Dictionary containing the data from the output files for each sector.
    """
    sector_specific_data = {}
    sectors_available = count_sector_files_in_output_dir(output_dir, tic_id)
    
    for i in range(sectors_available):
        with h5py.File(f'{tic_id}_sector_combined.h5', 'r') as h5:
            for sector_name in h5.keys():  # Explicitly use .keys() to iterate over all groups
                sector_group = h5[sector_name] 
                sector_specific_data[sector_name] = {}
                for dset_name in sector_group['flares'].keys():
                    dset = sector_group['flares'][dset_name][:]
                    sector_specific_data[sector_name][dset_name]= dset
                
                for dset_name in sector_group['lc'].keys():
                    dset = sector_group['lc'][dset_name][:]
                    sector_specific_data[sector_name][dset_name]= dset
                
                for dset_name in sector_group['injections'].keys():
                    dset = sector_group['injections'][dset_name][:]
                    sector_specific_data[sector_name][dset_name]= dset
            
                sector_recovered= sector_group['recovered'][:]
                sector_specific_data[sector_name]['recovered']= sector_recovered
    
    return sector_specific_data


def combine_sector_files(input_file_paths, output_file_name):
    """
    Combine multiple HDF5 files into a single file with each file's datasets in its own group.
    
    Parameters:
    - input_files: List of paths to the input HDF5 files.
    - output_file: Path to the output HDF5 file.
    """

    def copy_group(src, dest):
        """
        Recursively copy all contents from src group to dest group.
        """
        for name, item in src.items():
            if isinstance(item, h5py.Dataset):
                dest.copy(item, name)
            elif isinstance(item, h5py.Group):
                new_group = dest.create_group(name)
                copy_group(item, new_group)

    if os.path.exists(output_file_name):
        return

    with h5py.File(output_file_name, 'w') as h5_out:
        for file_path in input_file_paths:
            # Use the file name (without extension) as the group name
            group_name = 'Sector'+os.path.splitext(os.path.basename(file_path))[0][11:]
            with h5py.File(file_path, 'r') as h5_in:
                # Create a new group in the output file for this input file
                new_group = h5_out.create_group(group_name)
                # Copy all contents from the input file to the new group
                copy_group(h5_in, new_group)


def calculate_eds_of_injected_flares(amps, fwhms):
    """
    Calculate the equivalent durations of the injected flares.
    
    Parameters:
    - amps: Array of flare amplitudes
    - fwhms: Array of flare FWHMs
    
    Returns:
    - injected_eds: Array of the equivalent durations of the injected flares.
    """
    # Need a dummy time array to calculate the equivalent duration
    # Step size is mapped to TESS 2 minute cadence
    dummy_time=np.arange(0,27, step=0.00138886)
    
    injected_eds=np.zeros(len(amps))

    for i in range(len(amps)):
        model_flux=np.nan_to_num(lupita.flare_model(dummy_time, 10, fwhms[i],
                                      amps[i]), nan=0.0)
    
        injected_eds[i]=(trapezoid(x=dummy_time[7000:8000], y=model_flux[7000:8000])* u.day ).to(u.s).value

    return injected_eds

def calculate_completeness(recovered):
    """
    Calculate the completeness of the injected flares.
    
    Parameters:
    - recovered: Array of booleans indicating whether each flare was recovered
    
    Returns:
    - completeness: 1D array of the fraction of injected flares that were recovered 
    """

    completeness=(np.sum(recovered, axis=0)/len(recovered))

    return completeness 

def Logistic_function(x,k,x0):
    y = 0.96 / (1 + np.exp(-k*(x-x0)))
    
    return y

def fit_completeness_function(completeness, recovered, injected_eds):
    """
    Fit a completeness curve to the calculated completeness values.
    
    Parameters:
    - completeness: fraction of injected flares that were recovered at a given ED.
    - injected_eds: Array of the equivalent durations of the injected flares.
    
    Returns:
    - fit: 1D array of the fitted completeness values.
    - fifty: The 50% completeness value
    """

    # determine jackknife errors
    jackknife_completeness = np.zeros(((len(completeness)), (len(recovered[0]))))
    jackknife_recovered_to_del = recovered

    for i in range(len(recovered[0])):
        jackknife_temp = np.delete(jackknife_recovered_to_del, i)
        jackknife_completeness[i] = calculate_completeness(jackknife_temp)

    jackknife_uncertainty = np.std(jackknife_completeness, axis=0)

    # Fit a logistic function completeness curve to the calculated completeness values
    popt, pcov= curve_fit(Logistic_function, injected_eds, completeness, sigma=jackknife_uncertainty)
    
    k_fit, x0_fit=popt
    x_interp=np.arange(min(injected_eds), max(injected_eds), 0.1)

    fit=Logistic_function(x_interp, k_fit, x0_fit)

    # Calculate the 50% completeness value
    fifty=np.interp(0.5, fit, x_interp)
    
    return fit, np.log10(fifty), k_fit, x0_fit #Set 50% comp to log to match the rest of the EDs

def calculate_raw_flare_rates(equivalent_durations, flare_starts, flare_ends, duration, lc_table):
    """
    Calculate the raw flare rates.
    
    Parameters:
    - equivalent_durations: Array of the equivalent durations of the injected flares.
    - flare_starts: Array of the indices of the start of each flare.
    - flare_ends: Array of the indices of the end of each flare.
    - duration: The duration of the light curve.
    - flux_err: The flux error of the light curve.
    
    Returns:
    - raw_rates: Array of the raw flare rates.
    """

    # Calculate Flare Durations
    flare_durations=np.zeros(len(flare_starts))

    for i in range(len(flare_starts)):
        flare_ind=range(flare_starts[i], flare_ends[i])

        flare_durations[i]= lc_table['time'][flare_ind][-1] - lc_table['time'][flare_ind][0]

    # Calculate the raw flare rates using Jim's code
    ed,rate,ed_err,rate_err = FFD(equivalent_durations, dur=flare_durations, Lum=0, TOTEXP=duration,
                    fluxerr=np.median(np.array(lc_table['flux_err']))/np.median(np.array(lc_table['flux'])))    

    
    return ed,rate,ed_err,rate_err #ED is in log seconds 

def perform_completeness_correction(comp_logistic_function_params, FFD_eds,
                                    fifty_percent_complentess_limit, raw_rates, rate_err):
    """
    Perform a completeness correction on the raw flare rates.
    
    Parameters:
    - comp_logistic_function_params: Array of the logistic function parameters.
    - FFD_eds: Array of the equivalent durations of the FFD.
    - fifty: The 50% completeness value
    - raw_rates: Array of the raw flare rates.
    - rate_err: Array of the error in the raw flare rates.
    
    Returns:
    - corrected_rates: Array of the completeness-corrected flare rates.
    """

    # Calculate the completeness correction
    comp = Logistic_function(10**FFD_eds, comp_logistic_function_params[0], comp_logistic_function_params[1])

    if len(raw_rates[FFD_eds > fifty_percent_complentess_limit]) > 0:
        corrected_rates = (raw_rates[FFD_eds > fifty_percent_complentess_limit]-np.log10(comp[FFD_eds > fifty_percent_complentess_limit]))
        eds_above_lim = FFD_eds[FFD_eds > fifty_percent_complentess_limit]
        rates_above_lim = rate_err[FFD_eds > fifty_percent_complentess_limit]

        return corrected_rates, eds_above_lim, rates_above_lim
    
    else:
        return np.array([0]), np.array([0]), np.array([0])

def line(x, a, b):
    return -a*x + b

# Define a wrapper function to fix the slope
def line_with_fixed_slope(x, fixed_slope, intercept):
    return line(x, fixed_slope, intercept)

def combine_all_sectors_and_fit_FFD(list_of_sector_flare_starts, list_of_sector_flare_ends, eds, sector_durations, lc_tables,
                                    fifty_percent_complentess_limits, comp_function_params):
    """
    Combine the flares and EDs from all sectors into a single array to fit FFD.

    Parameters:
    - list_of_sector_flare_starts: list of each sectors flare starts
    - list_of_sector_flare_ends: list of each sectors flare ends
    - eds: list of each sectors equivalent durations
    - sector_durations: list of each sectors duration
    - lc_tables: list of each sectors light curve tables
    - fifty_percent_complentess_limits: list of each sectors 50% completeness limits
    - comp_function_params: list of each sectors logistic function parameters

    Returns:
    - slope: Best fit Power-Law slope of the FFD for combined sample
    - intercept: Best fit Power-Law intercept of the FFD for combined sample
    """

    if not eds:  # Check if eds is empty
        return -99, -99, -99
    # Combine the EDs from all sectors into a single array
    all_eds = np.concatenate(eds)
    
    all_times=[]
    all_fluxes=[]
    all_flux_err=[]
    for i in range(len(lc_tables)):
        for j in range(len(lc_tables[i])):
            all_times.append(lc_tables[i]['time'][j])
            all_fluxes.append(lc_tables[i]['flux'][j])
            all_flux_err.append(lc_tables[i]['flux_err'][j])

    tot_duration = np.sum(sector_durations)

    flare_durations=[]

    for i in range(len(lc_tables)):
        for j in range(len(list_of_sector_flare_starts[i])):
            flare_ind=range(list_of_sector_flare_starts[i][j], list_of_sector_flare_ends[i][j])

            flare_durations.append(lc_tables[i]['time'][flare_ind][-1] - lc_tables[i]['time'][flare_ind][0])

    # Fit the FFD
    ffd_ed, raw_rate, ed_err, rate_err = FFD(all_eds, dur=flare_durations, Lum=0, TOTEXP=tot_duration, 
                                            fluxerr=np.median(np.array(all_flux_err))/np.median(np.array(all_fluxes)))
    
    # Perform a completeness correction on the FFD
    corrected_rate, eds_above_lim, rate_err_above_lim = perform_completeness_correction(np.median(comp_function_params, axis=0), ffd_ed,
                                                                    np.median(fifty_percent_complentess_limits),
                                                                    raw_rate, rate_err)
    

    if corrected_rate[0] == 0:
        return -99, -99, all_eds
    # Fit a power-law to the FFD

    params_all_sec_combined_corrected, var_all_sec_combined_corrected = curve_fit(line, eds_above_lim, corrected_rate, sigma=rate_err_above_lim)

    slope = params_all_sec_combined_corrected[0]

    intercept = params_all_sec_combined_corrected[1] 

    return slope, intercept, all_eds

def fit_FFD_with_fixed_slope(corrected_rates, rate_err, corrected_eds, slope):
    """
    Fit the FFD for each Sector using the fixed slope from the full sample
    
    Parameters:
    - corrected_rates: Array of the completeness-corrected flare rates.
    - rate_err: Array of the error in the completeness-corrected flare rates.
    - corrected_eds: Array of the equivalent durations of the injected flares.
    - slope: The fixed slope from the full sample.
    
    Returns:
    - beta: The best fit Power-Law intercept of the FFD for the sector.
    """
    # Fit the FFD for each sector using the fixed slope from the full sample
    if corrected_rates[0] == 0:
        return -99, -99
    else:
        # Use the wrapper function with the fixed slope
        params, var = curve_fit(lambda x, intercept: line_with_fixed_slope(x, slope, intercept), 
                                corrected_eds, corrected_rates, sigma=rate_err)

        beta = params[0]

        return beta, var[0][0]

def make_diagnostic_plot(tic_id, lc_tables, betas, beta_errs, corrected_eds, corrected_rates, FFD_eds, 
                         raw_rates, F_flare_F_bols, fifty_percent_complentess_limits, output_path, 
                         full_sample_slope, show):
    """
    Make 3 Panel Diagnostic plot for each star containing 1) The Completeness Corrected FFD with best fit power law,
    2) The beta vs time plot and 3) L_flare_L_bol plot

    Parameters:
    - tic_id : `int'
        TIC ID for star 
    - lc_tables : `list'
        list of lc_tables 
    - betas : `list'
        list of sector beta values derived from fixed slope power law fit to FFD
    - FFD_eds :  `list'
        The FFD EDs
    - raw_rates : `list'
        Completeness corrected Flare rates
    - raw_rate_errors : `list'
        Uncertainties in the Flare rates
    - F_flare_F_bols : `list'
        Total integrated flare power per sector
    - fifty_percent_complentess_limits : `list'
         50% completeness limit per sector
    - output_path : `string' 
        where to put the saved figure
    - full_sample_slope : `int'
        Best fit power law slope for all sectors combined
    - show : `bool`, optional
        Whether to show the plot, by default True

    Returns:
    - fig :class:`matplotlib.figure.Figure`
        3 panel figure
    """
    # make the blue to red colormap
    max_time = float(0.)
    for table in lc_tables:
        table_max_time = max(table['time'])
        if table_max_time > max_time:
            max_time = table_max_time

    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])
    cnorm = mcol.Normalize(vmin=lc_tables[0]['time'][0],vmax=max_time)
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])


    fig = plt.figure()
    ax1 = plt.subplot(2,2,1)
    ax1 = plt.gca()
    ax1.cla()

    # First subplot is the FFD, loop over the sectors
    for i in range(len(lc_tables)): 
        # get x values for
        if betas[i] == -99:
            continue 
        PL_line_xs=np.arange(fifty_percent_complentess_limits[i], max(corrected_eds[i]), .01)
        ax1.scatter(FFD_eds[i], raw_rates[i], c=cpick.to_rgba(np.nanmedian(lc_tables[i]['time'])), 
                    s=15, alpha=.1)
        ax1.scatter(corrected_eds[i], corrected_rates[i], c=cpick.to_rgba(np.nanmedian(lc_tables[i]['time'])), 
                    s=35, marker = 's', alpha=.5)
        
        ax1.plot(corrected_eds[i], line(corrected_eds[i], full_sample_slope, betas[i]), 
                 c=cpick.to_rgba(np.nanmedian(lc_tables[i]['time'])), alpha=0.5)
        
    #Dummy lines for legend
    ax1.scatter(0,0, s=10, marker = 's', alpha=.75, label='Completeness \nCorrected', c='k')
    ax1.errorbar(0,0, yerr=.01, lw=0.5, capsize=2, alpha=.3, fmt='o', label='Raw', c='k')
    ax1.plot([0,0], [0,0], alpha=0.7, color='k', label='Powerlaw Fit')
    ax1.text(-0.5, -2.25, f'Slope: {full_sample_slope:.2f}', fontsize=20)
        
    ax1.set_xlabel('log Equivalent Duration (s)')
    ax1.set_ylabel('log Flare Rate (per day)')  
    ax1.set_xlim(-0.8,3.1)
    ax1.legend()

    # Second Figure is the Beta values vs time
    plot_sector_times = []
    for i in range(len(lc_tables)):
        plot_sector_times.append(np.nanmedian(lc_tables[i]['time']))

    ax2 = plt.subplot(2,2,2)
    ax2 = plt.gca()
    ax2.cla()

    if len(np.array(betas)[np.array(betas) > 0]) > 0:
        valid_inds=np.array(betas) > 0
        times = np.array(plot_sector_times)[valid_inds]
        valid_betas = np.array(betas)[valid_inds]
        valid_beta_errs = np.array(beta_errs)[valid_inds]
        colors = cpick.to_rgba(times)

        for i in range(len(valid_beta_errs)):
            if np.isfinite(valid_beta_errs[i]):
                valid_beta_errs[i] = valid_beta_errs[i] + 0.25
            else:
                valid_beta_errs[i] = 1.2

        for time, beta, color, yerr in zip(times, valid_betas, colors, valid_beta_errs):
            ax2.errorbar(time, beta, yerr=yerr, fmt='o', color=color, lw=1., capsize=2.5, linestyle='none')

    else:
        ax2.scatter(0, 0)
    ax2.set_ylabel(r'$\beta$')
    ax2.set_xlabel('Time')

    # Third Figure is the L_flare_L_bol vs time
    ax3 = plt.subplot(2,2,3)
    ax3 = plt.gca()
    ax3.cla()

    ax3.scatter(plot_sector_times, np.log10(np.array(F_flare_F_bols)), c=cpick.to_rgba(np.array(plot_sector_times)), s=100)
    ax3.set_ylabel(r'$log(\frac{L_{fl}}{L_{TESS}})$', fontsize=30)
    ax3.set_xlabel('Time')

    ax4 = plt.subplot(2,2,4)
    ax4 = plt.gca()
    ax4.cla()

    # Straight flares per sector Time on x-axis 
    flares_per_sector = []
    for i in range(len(lc_tables)):
        flares_per_sector.append(len(FFD_eds[i]))

    ax4.scatter(plot_sector_times, flares_per_sector, c=cpick.to_rgba(np.array(plot_sector_times)), s=100)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('N Flares')

    plt.subplots_adjust(right=2., top=2.3)
    plt.savefig(output_path+f'{tic_id}_Diagnostic_Figure.pdf', dpi=250, bbox_inches='tight', 
                pad_inches=0.05, facecolor='w', format='pdf')

    if show:
        plt.show()

    else:
        plt.close(fig)

    return fig

def post_process(tic_id, output_dir, out_path, show=False):
    
    # Step 1: Count the number of sector files available for the given TIC ID
    # Then Merge the sector files into a single file

    # setup the logger with output
    logger = logging.getLogger("flair")
    logger.setLevel('NOTSET')

    sectors_available = count_sector_files_in_output_dir(output_dir, tic_id)

    list_of_file_names = []
    for i in range(sectors_available):
        list_of_file_names.append(f'{output_dir}{tic_id}_{i}.h5')

    combine_sector_files(list_of_file_names, f'{output_dir}{tic_id}_sector_combined.h5')

    logger.info('Combined all sectors into single file')

    # Step 2: Get the sector specific data in accessible format
    sector_specific_data = access_data(tic_id, output_dir)

    # Step 3: Read out required sector specific data to lists to easily access
    eds = []
    recovered = []
    flare_starts = []
    flare_ends = []

    injected_amps = []
    injected_fwhms = []

    for sector in sector_specific_data:
        eds.append(sector_specific_data[sector]['equivalent_durations'])
        recovered.append(sector_specific_data[sector]['recovered'])
        flare_starts.append(sector_specific_data[sector]['start_times'])
        flare_ends.append(sector_specific_data[sector]['stop_times'])
        injected_amps.append(sector_specific_data[sector]['amps'])
        injected_fwhms.append(sector_specific_data[sector]['fwhms'])

    # Step 4: Calculate the completeness of the injected flares per sector
    completeness = []
    for i in range(len(recovered)):
        completeness.append(calculate_completeness(recovered[i]))

    logger.info("Completeness is calculated")

    # Step 5: Calculate the injected EDs per sector
    injected_eds = []
    for i in range(len(injected_amps)):
        injected_eds.append(calculate_eds_of_injected_flares(injected_amps[i], injected_fwhms[i]))

    # Step 6: Fit a completeness function to the calculated completeness values per sector
    completeness_fits = []
    fifty_percent_complentess_limits = []
    comp_function_params=np.zeros((len(completeness), 2))

    for i in range(len(completeness)):
        fit, fifty, k_fit, x0_fit = fit_completeness_function(completeness[i], recovered[i], injected_eds[i])
        completeness_fits.append(fit)
        fifty_percent_complentess_limits.append(fifty)
        comp_function_params[i] = [k_fit, x0_fit]

    # step 7 Derive the sector durations
    sector_durations = np.zeros(sectors_available)
    sector_lcs = []
    for sector in sector_specific_data:
        lc_table = Table([sector_specific_data[sector]['time'], sector_specific_data[sector]['flux'],
                       sector_specific_data[sector]['flux_err']],
                        names=('time', 'flux', 'flux_err') )
        sector_lcs.append(lc_table)

    for i in range(sectors_available):
        sector_durations[i]=sector_lcs[i]['time'][-1]-sector_lcs[i]['time'][0]


    # Step 8: Calculate the raw flare rates per sector
    raw_rates = []
    FFD_eds = []
    raw_rate_err = []
    FFD_ed_err = []
    for i in range(len(eds)):
        ed,rate,ed_err,rate_err = calculate_raw_flare_rates(eds[i], flare_starts[i], 
                                                            flare_ends[i], sector_durations[i], sector_lcs[i])
        raw_rates.append(rate)
        FFD_eds.append(ed)
        raw_rate_err.append(rate_err)
        FFD_ed_err.append(ed_err)

    # Step 9: Perform a completeness correction on the raw flare rates per sector
    corrected_rates = []
    corrected_eds = []
    rate_err_above_lim = []
    for i in range(len(raw_rates)):
        cr, c_eds, c_err = perform_completeness_correction(comp_function_params[i], FFD_eds[i],
                                                fifty_percent_complentess_limits[i], raw_rates[i], 
                                                raw_rate_err[i])
        corrected_rates.append(cr)
        corrected_eds.append(c_eds)
        rate_err_above_lim.append(c_err)

    logger.info("FFD has been calculated and corrected")

    # Step 10: Combine the flares and EDs from all sectors into a single array to fit FFD 
    full_sample_slope, full_sample_intercept, all_eds = combine_all_sectors_and_fit_FFD(flare_starts, flare_ends, eds, 
                                                                               sector_durations, sector_lcs,
                                                                               fifty_percent_complentess_limits,
                                                                               comp_function_params)

    logger.info(f"Full sample FFD has been fit with a bestfit slope of {full_sample_slope}")

    # Step 11: Fit the FFD for each Sector using the fixed slope from the full sample
    sector_beta_values = []
    sector_beta_err_values = []

    if full_sample_slope == -99:
        logger.error("No Flares were found in the full sample, so no FFD was fit")
        sector_beta_values = [-99 for i in range(sectors_available)]
        sector_beta_err_values = [-99 for i in range(sectors_available)]
    else:
        for i in range(sectors_available):
            beta, beta_err = fit_FFD_with_fixed_slope(corrected_rates[i], rate_err_above_lim[i], corrected_eds[i], full_sample_slope)
            sector_beta_values.append(beta)
            sector_beta_err_values.append(beta_err)

    logger.info("The Sector FFD's have now been fit with a Power-law and the intercepts have been saved")
    
    # Step 12: Calculate other Flare metrics
    F_Flare_F_Bol = []

    for i in range(sectors_available):
        sd_in_s= sector_durations[i]*u.day.to(u.s)
        F_Flare_F_Bol.append(np.sum(eds[i])/sd_in_s)


    # Step 13: Make Diagnostic plot
    fig_save_path = '../Final_Outputs/diagnostic_plots/'

    if full_sample_slope == -99:
        fig = "No Flares were found in the full sample, so no FFD was fit"
    else:
        fig = make_diagnostic_plot(tic_id, sector_lcs, sector_beta_values, sector_beta_err_values, 
                                   corrected_eds, corrected_rates, FFD_eds, raw_rates,
                                   F_Flare_F_Bol, fifty_percent_complentess_limits,
                                   fig_save_path, full_sample_slope, show)
    
    logger.info(f'Diagnostic Figure is made, you should go check it out in {fig_save_path}')

    # Step 14: Write out what we actually want to a new h5 file

    # setup the file name and check if it already exists
    file_name = out_path + f"{tic_id}_final_output.h5"

    with h5py.File(file_name, 'w') as h5:
        dt = h5py.special_dtype(vlen=np.dtype('float64'))
        g = h5.create_group('flares')
        # Create dataset with variable-length data type
        dset = g.create_dataset('flare_starts', (len(flare_starts),), dtype=dt)
        # Write each sublist to the dataset
        for i, sublist in enumerate(flare_starts):
            dset[i] = sublist
        
        dset = g.create_dataset('flare_ends', (len(flare_ends),), dtype=dt)
        # Write each sublist to the dataset
        for i, sublist in enumerate(flare_ends):
            dset[i] = sublist

        source_file_path = f'{output_dir}{tic_id}_sector_combined.h5'
        sector_flare_masks = []
        with h5py.File(source_file_path, 'r') as source_file:
            for sector_name in source_file.keys():
                sector_flare_masks.append(source_file[sector_name]['lc']['flare_mask'][:])
        
        dset = g.create_dataset('flare_masks', (len(sector_flare_masks),), dtype=dt)
        # Write each sublist to the dataset
        for i, sublist in enumerate(sector_flare_masks):
            dset[i] = sublist
                
        # Create a group named 'lcs'
        tables_group = h5.create_group('lcs')
        for i, table in enumerate(sector_lcs):
            dataset_name = f'sector_{i}_lc'
            table.write(tables_group, path=dataset_name, format='hdf5')

        g_f = h5.create_group('FFD')
        dset = g_f.create_dataset('FFD_eds', (len(FFD_eds),), dtype=dt)
        for i, sublist in enumerate(FFD_eds):
            dset[i] = sublist

        dset = g_f.create_dataset('raw_rates', (len(raw_rates),), dtype=dt)
        for i, sublist in enumerate(raw_rates):
            dset[i] = sublist

        dset = g_f.create_dataset('raw_rate_err', (len(raw_rate_err),), dtype=dt)
        for i, sublist in enumerate(raw_rate_err):
            dset[i] = sublist

        dset = g_f.create_dataset('corrected_rates', (len(corrected_rates),), dtype=dt)
        for i, sublist in enumerate(corrected_rates):
            dset[i] = sublist

        dset = g_f.create_dataset('corrected_eds', (len(corrected_eds),), dtype=dt)
        for i, sublist in enumerate(corrected_eds):
            dset[i] = sublist

        dset = g_f.create_dataset('rate_err_above_lim', (len(rate_err_above_lim),), dtype=dt)
        for i, sublist in enumerate(rate_err_above_lim):
            dset[i] = sublist
        g_f.create_dataset('full_sample_slope', data=full_sample_slope)

        g_m = h5.create_group('metrics')
        g_m.create_dataset('fifty_percent_completeness_limits', data=fifty_percent_complentess_limits)
        g_m.create_dataset('comp_function_params', data=comp_function_params)
        g_m.create_dataset('sector_beta_values', data=sector_beta_values)
        g_m.create_dataset('F_Flare_F_Bol', data=F_Flare_F_Bol)


        injection_recovery_group = h5.create_group('Injection_Recovery')
        for i in range(len(recovered)):
            sector_group = injection_recovery_group.create_group(f'sector_{i}')
            sector_group.create_dataset('recovered', data=recovered[i])
            sector_group.create_dataset('injected_eds', data=injected_eds[i])
            sector_group.create_dataset('injected_amps', data=injected_amps[i])

        # Open the source HDF5 file in read mode
        gp_group = h5.create_group('GP')
        with h5py.File(source_file_path, 'r') as source_file:
            # Iterate through the sector groups in the source file
            for sector_name in source_file.keys():
                actual_name = sector_name.split('_')[1]
                sector_group = source_file[sector_name]
                gp_ = sector_group['gp']
                # Copy the 'GP' subgroup to the corresponding sector in the destination file
                subsector_group = gp_group.create_group('sector_'+actual_name)
                source_file.copy(gp_, subsector_group, name='GP')

    logger.info(f"Final output file has been written out to {file_name}")

    print(f"All done with {tic_id}! Check out the final output file for the results")

   


# setup argparse
def main():
    parser = argparse.ArgumentParser(description='Run Post processing pipeline on a a CVZ star')
    parser.add_argument('-t', '--tic', default="", type=str,
                        help='TIC ID of the star')
    
    parser.add_argument('-p', '--path', default=".", type=str,
                        help='Output path of the data to read in')
    
    parser.add_argument('-s', '--show', default=0, type=int,
                        help='Show plot?')

    args = parser.parse_args()

    # run the pipeline
    post_process(tic_id=args.tic, output_dir=args.path, show=args.show==1)

if __name__ == "__main__":
    main()