import stella
import numpy as np


def prep_stella(out_path='data'):
    cnn = stella.ConvNN(output_dir=out_path)

    ds = stella.download_nn_set.DownloadSets()
    ds.download_models()

    return cnn, ds.models


def get_stella_predictions(cnn, models, lc):
    # predict the flare probability for each model
    preds = [cnn.predict(modelname=model, times=lc.time.value,
                         fluxes=lc.flux.value, errs=lc.flux_err.value)[0]
             for model in models]

    # return the median probability across all models, ignoring NaNs
    return np.nanmedian(preds, axis=0)