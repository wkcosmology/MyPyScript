from scipy.spatial import KDTree
import numpy as np


class LOESS2D:
    def __init__(self, x, y, val, n_nbr, boxsize=None, xy_ratio=1):
        """
        x, y, val: ndarray with shape (N, )
            input coordinate and values
        n_nbr: number of neighbours for smoothing, or fraction w.r.t. to the total population
            # of neighbors = (n_nbr >= 1) ? int(n_nbr) : int(n_nbr * N)
        boxsize: optional
            if assigned a value, the distance is calculated in a periodic box
        xy_ratio:
            weight in the calculation of distance
            d = sqrt(xy_ratio * (x_1 - x_2)^2 + (y_1 - y_2)^2)^2
        """
        # Record the transformation for x and y coordinates
        self._xnorm = self._gen_norm(x, xy_ratio)
        self._ynorm = self._gen_norm(y)
        self._xn = self._xnorm(x)
        self._yn = self._ynorm(y)
        self._val = val.copy()
        self._tree = KDTree(
            np.column_stack((self._xn, self._yn)), copy_data=True, boxsize=boxsize
        )
        if n_nbr >= 1:
            self._n_nbr = int(n_nbr)
        else:
            self._n_nbr = int(frac * len(x))
        if self._n_nbr > len(x):
            raise Exception(
                "Number of smoothing neighbors exceeds the total number of points"
            )
        print("# of neightbours for smoothing: %d" % self._n_nbr)

    def __call__(self, x, y):
        x_norm = self._xnorm(x)
        y_norm = self._ynorm(y)
        d_nbr, i_nbr = self._tree.query(np.column_stack((x_norm, y_norm)), self._n_nbr)
        d_norm = (d_nbr.T / d_nbr[:, -1]).T
        weight = (1 - d_norm**3) ** 3
        val = np.sum(weight * self._val[i_nbr], axis=1) / np.sum(weight, axis=1)
        return val

    def _gen_norm(self, arr, ratio=1):
        """
        Normalize the coordinate using quantiles rather than the standard deviation
        to avoid the impact of outliners.
        """
        xl, x_med, xu = np.quantile(arr, [0.17, 0.5, 0.84])
        return lambda x: (arr - x_med) / (xu - xl) * ratio

if __name__ == "__main__":
    gal = gals_sdss.copy()
    sm = gal["sm"].values
    hm = gal["hm"].values
    ssfr = gal["ssfr"].values
    loess = LOESS2D(sm, hm, ssfr, 30)
    ssfr_smooth = loess(sm, hm)

    plt.subplots(1, 1, figsize=(15, 10))
    # Get the black edges
    # rasterized=True is important, otherwise you will get a very large figure file
    plt.scatter(*gal["sm hm".split()].values.T, lw=5, s=80, rasterized=True)
    # fill the color
    bar = plt.scatter(*gal["sm hm".split()].values.T, c=ssfr_smooth, lw=0, s=80, cmap='Spectral', rasterized=True, vmax=-9.5, vmin=-12.5)
    plt.colorbar(bar, label=r"$\rm \log~SSFR$")
    plt.ylim(11.5, 16)
    plt.xlim(8.5, 12.5)
    plt.xlabel(r"$\log M_*$")
    plt.ylabel(r"$\log M_h$")
