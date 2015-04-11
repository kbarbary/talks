import numpy as np

ccm89_coeffs_a = np.array([0.32999, -0.77530, 0.01979, 0.72085, -0.02427,
                           -0.50447, 0.17699, 1.])
ccm89_coeffs_b = np.array([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233,
                            2.28305, 1.41338, 0.])

def ccm89(wave, r_v):

    x = 1.e4 / wave
    if np.any(x < 0.3) or np.any(x > 11.):
        raise ValueError('CCM law valid only for wavelengths from '
                         '910 Angstroms to 3.3 microns')

    # Initialize output arrays.
    a = np.empty_like(x)
    b = np.empty_like(x)

    # Near Infrared.
    mask = (0.3 <= x) & (x < 1.1)
    a[mask] = 0.574 * x[mask]**1.61
    b[mask] = -0.527 * x[mask]**1.61

    # Optical.
    mask = (1.1 <= x) & (x < 3.3)
    y = x[mask] - 1.82
    a[mask] = np.polyval(ccm89_coeffs_a, y)
    b[mask] = np.polyval(ccm89_coeffs_b, y)

    # Ultraviolet.
    mask = (3.3 <= x) & (x < 8.)
    y = x[mask]
    f_a = np.zeros_like(y)
    f_b = np.zeros_like(y)
    select = (y >= 5.9)
    yselect = y[select] - 5.9
    f_a[select] = -0.04473 * yselect**2 - 0.009779 * yselect**3
    f_b[select] = 0.2130 * yselect**2 + 0.1207 * yselect**3
    a[mask] = 1.752 - 0.316*y - (0.104 / ((y-4.67)**2 + 0.341)) + f_a
    b[mask] = -3.090 + 1.825*y + (1.206 / ((y-4.62)**2 + 0.263)) + f_b

    # Far-UV (CCM89 extrapolation)
    mask = (8. <= x) & (x < 11.)
    y = x[mask] - 8.
    coef_a = np.array([-0.070, 0.137, -0.628, -1.073])
    coef_b = np.array([0.374, -0.420, 4.257, 13.670])
    a[mask] = np.polyval(coef_a, y)
    b[mask] = np.polyval(coef_b, y)

    return a + b / r_v
