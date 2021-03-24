import numpy as np


def n2o_forcing_AR6(n2o, n2o_0, co2_bar, ch4_bar):

    a2 = -8.0e-6  # W/m2/ppb
    b2 = 4.2e-6   # W/m2/ppb
    c2 = -4.9e-6  # W/m2/ppb
    n2o_bar = (n2o_0+n2o)/2.
    n2o_forcing_AR6 = (a2*co2_bar+b2*n2o_bar + c2*ch4_bar + 0.117) * \
        (np.sqrt(n2o)-np.sqrt(n2o_0))
    return n2o_forcing_AR6
