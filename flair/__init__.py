import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # suppress tensorflow warnings
from . import flares, gp, inject, jim, lightcurve, lupita, plot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'            # re-enable tensorflow warnings
del os
from ._version import __version__