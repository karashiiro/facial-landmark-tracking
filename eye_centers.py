import numpy as np

# 16x16 => 18ms
# 32x32 => 86ms
# 64x64 => 730ms
EYE_SIZE = (16, 16)

def objective(w, gradient, grid, c):
    displacements = np.subtract(grid, c)
    magnitudes = np.sqrt(np.sum(np.square(displacements), axis=2))
    # The epsilon efficiently accounts for magnitude == 0; we would otherwise need
    # to use np.where which is comparatively slow.
    units = np.divide(displacements, np.dstack([magnitudes, magnitudes]) + 0.001)
    w_dg_sqr = np.square(np.tensordot(units, gradient, axes=([0, 1], [0, 1]))) * w
    w_dg_sqr_sum = np.sum(w_dg_sqr.flatten())
    return w_dg_sqr_sum / w_dg_sqr_sum.size

def get_centre(weights: np.ndarray, gradient: np.ndarray) -> list:
    """
    Predicts eye centre position based on the provided gradient and
    weight map. Implementation based on `Timm, F. and Barth, E. (2011).
    "Accurate eye centre localisation by means of gradients".`
    """
    grid = np.array([[[x, y] \
        for y in range(EYE_SIZE[1])] \
            for x in range(EYE_SIZE[0])])
    obj_results = np.array([objective(w, gradient, grid, c) for c, w in np.ndenumerate(weights)])
    grid_flat = grid.reshape((EYE_SIZE[0] * EYE_SIZE[1], 2))
    return grid_flat[np.argmax(obj_results)]
