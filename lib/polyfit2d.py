import numpy as np


def polyfit2d(regionsList, data, kx, ky, order=None):
    """ 
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    regionsList: list
        region - lisk like [(x1,y1),(x2,y2)]
        where x1, x2, y1, y2 good ordered coordinates (x2 > x1 and y2 > y1).
    data: np.ndarray, 2d
        Surface with regions to fit.
    kx, ky: int,
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.
    """
    z = []
    x = []
    y = []
    for region in regionsList:
        x1, y1 = region[0]
        x2, y2 = region[1]
        z.extend(list(np.ravel(data[y1:y2, x1:x2])))
        # z.extend(list(np.ravel(data[x1:x2,y1:y2])))
        xReg = range(x1, x2)
        yReg = range(y1, y2)
        xReg, yReg = np.meshgrid(xReg, yReg)
        x.extend(list(np.ravel(xReg)))
        y.extend(list(np.ravel(yReg)))

    # 1d surfaces from regions
    z = np.array(z)

    # grid coords
    x = np.array(x)
    y = np.array(y)

    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx + 1, ky + 1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x ** i * y ** j
        a[index] = arr

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)
