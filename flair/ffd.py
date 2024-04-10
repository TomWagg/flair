import numpy as np
    

def sample_flare_energies(stellar_class, n_samples=1000, E_min=1e27, E_max=1e40):
    """Calculate the flare frequency for a given bolometric flare energy and stellar class.

    Parameters
    ----------
    stellar_class : `str`
        Stellar class (e.g. "M", "K", "G")
    n_samples : `int`
        Number of samples to draw
    E_min : `float`
        Minimum bolometric flare energy (erg)
    E_max : `float`
        Maximum bolometric flare energy (erg)

    Notes
    -----
    This function calculates the flare frequency using the relation from Howard+2019 for M and K dwarfs and
    the relation from Shibayama+2013 for G dwarfs. The relation is given by
        $$\log \nu = \alpha * \log E_{\rm bol} + \beta$$
    where $\nu$ is the cumulative flare frequency (per day) and $E_{\rm bol}$ is the bolometric flare energy.

    For M/K dwarfs, we assume active M2 and K5 dwarfs are representative of all respectively. For G dwarfs we
    convert the cumulative flare frequency to a flare frequency by multiplying by the bolometric flare energy
    and dividing by 365 days/year.

    For K dwarfs, we assume the flare frequency is intermediate to the M and G dwarf flare frequencies given
    the original relation from Howard+2019 was too steep at low flare energies.

    Returns
    -------
    e_bol : `np.ndarray`
        Sampled bolometric flare energies (erg)
    n_tot : `float`
        Average number of total flares in the given energy range (for normalisation)
    """
    # draw random samples uniformly
    u = np.random.rand(n_samples)

    # set the parameters for the given stellar class
    params = {
        "M": (-0.84, 26.82),
        "K": (-0.84, 25.3),
        "G": (-1.8, 25.6)
    }
    alpha, beta = params[stellar_class]

    # quick function to calculate the flare frequency
    nu = (lambda e_bol: 10**beta * e_bol**alpha) if stellar_class in ["M", "K"] else (lambda e_bol: 10**beta * e_bol**alpha * e_bol / 365)

    # total number of flares is just the difference
    n_tot = nu(E_min) - nu(E_max)

    # calculate the bolometric flare energies
    if stellar_class in ["M", "K"]:
        return (((1 - u) * n_tot) / 10**beta)**(1 / alpha), n_tot
    elif stellar_class == "G":
        return ((365 * (1 - u) * n_tot) / 10**beta)**(1 / (alpha + 1)), n_tot


# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()

# for stellar_class in ["M", "K", "G"]:
#     e_bols, n_tot = sample_flare_energies(stellar_class, 1000000)
#     print(n_tot)
#     bins = np.linspace(27, 36, 300)
#     weights = np.repeat(n_tot / len(e_bols), len(e_bols))

#     print(e_bols)

#     nu = np.zeros(len(bins))
#     for i in range(len(bins)):
#         nu[i] = np.sum(weights[e_bols > 10**bins[i]])

#     ax.plot(bins, nu, label=stellar_class)
    

# ax.legend()
# plt.yscale("log")
# ax.set(xlim=(28, 36), ylim=(1e-3, 1e2))
# plt.show()
