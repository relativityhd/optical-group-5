import numpy as np
import laspy as lp
import xarray as xr
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

def tin_interpolation(X:np.ndarray, Y: np.ndarray, Z: np.ndarray):
    assert X.shape == Y.shape == Z.shape, "X, Y, Z must have the same shape"

    # Filter out Z == nan (0)
    mask = ~np.isnan(Z)
    X_nozero = X[mask]
    Y_nozero = Y[mask]
    Z = Z[mask]

    # Create Delaunay triangulation
    tri = Delaunay(np.c_[X_nozero.ravel(), Y_nozero.ravel()])

    # Create interpolator
    f = LinearNDInterpolator(tri, Z.ravel())

    Z_interp = f(X, Y)

    return Z_interp

def point_height(points: np.ndarray, w: float = 10):
    # Generate raster coordinates from points in a vectorized form
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()

    # Generate raster coordinates
    x = np.arange(xmin + w / 2, xmax + w / 2, w)
    y = np.arange(ymin + w / 2, ymax + w / 2, w)

    # Map points to raster coordinates
    z = np.zeros((len(y), len(x)))
    ix = np.floor((points[:, 0] - xmin) / w).astype(int)
    iy = np.floor((points[:, 1] - ymin) / w).astype(int)

    # Vectorized version
    np.maximum.at(z, (iy, ix), points[:, 2])

    return x, y, z

def point_class(points: np.ndarray, classes: np.ndarray, w: float = 10):
    # Generate raster coordinates from points in a vectorized form
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()

    # Generate raster coordinates
    x = np.arange(xmin + w / 2, xmax + w / 2, w)
    y = np.arange(ymin + w / 2, ymax + w / 2, w)

    # Map points to raster coordinates
    z = np.zeros((len(y), len(x), classes.max() + 1), dtype=int)
    ix = np.floor((points[:, 0] - xmin) / w).astype(int)
    iy = np.floor((points[:, 1] - ymin) / w).astype(int)

    # Get the most frequent class in each cell of the raster
    np.add.at(z, (iy, ix, classes), 1)

    # Compute the most frequent class for each cell
    z = z.argmax(axis=-1)

    return x, y, z

def classes(f: lp.LasData, res: float = 1, subsampling: int = None):
    p = np.c_[f.x, f.y, f.z]
    c = f.classification
    if subsampling:
        p = p[::subsampling]
        c = c[::subsampling]

    x, y, Z = point_class(p, c, w=res)

    da = xr.DataArray(Z, coords=[y, x], dims=['lat', 'lon'])
    return da


def dem(f: lp.LasData, res: float = 1, subsampling: int = None, interp: bool = False, keep_class: int = None):
    p = np.c_[f.x, f.y, f.z]
    if subsampling:
        p = p[::subsampling]

    if keep_class:
        p = p[f.classification == keep_class]

    x, y, Z = point_height(p, w=res)
    if interp:
        X, Y = np.meshgrid(x, y)
        mask = Z != 0
        Z = np.where(mask, Z, np.nan)
        Z = tin_interpolation(X, Y, Z)

    da = xr.DataArray(Z, coords=[y, x], dims=['lat', 'lon'])
    return da
