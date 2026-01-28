"""
Defines a "kspec" class for making 2-D angular power spectra of CMB maps (CAR pixellization)
"""

from pixell import enmap, enplot
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import pickle
import itertools
from astropy import wcs


def get_fsky(window: enmap.ndmap):
    pixsize_map = window.pixsizemap()
    w2 = np.sum(window**2 * pixsize_map)
    w4 = np.sum(window**4 * pixsize_map)
    Omega = w2**2 / w4
    return Omega / (4 * np.pi)


def test_same_geometry(enmap_1, enmap_2):
    return enmap_1.shape == enmap_2.shape  # TODO : also check wcs


def trim_enmap_at_ell(
    lx: np.ndarray,
    ly: np.ndarray,
    map_to_trim: enmap.ndmap,
    ell_trim: float,
    shift_lx: float = 0,
    shift_ly: float = 0,
):  # TODO : make this also work for non-shifted maps ?
    """
    Trim a given fftshifted ndmap (1D or 3D) to ell_trim using lx and ly arrays associated to the map.
    Optionnaly shift the trim using shift_lx and shift_ly.
    """
    assert ell_trim > 0

    idx = (lx < ell_trim + shift_lx) & (lx > -ell_trim + shift_lx)
    idy = (ly < ell_trim + shift_ly) & (ly > -ell_trim + shift_ly)
    mask_2d = np.ix_(idy, idx)
    map_trimmed = map_to_trim[mask_2d]
    return map_trimmed


TQU_indices: dict[str, int] = {
    "T": 0,
    "Q": 1,
    "U": 2,
    "E": 0,
    "B": 1,
}


class So_Spec2D:
    # self.ncomp = None
    shape = None
    wcs = None
    ncomp = None
    Nsplits = None
    lx: np.ndarray = None
    ly: np.ndarray = None
    lxmap: enmap.ndmap = None
    lymap: enmap.ndmap = None
    llims: tuple = None
    thetamap: enmap.ndmap = None
    modlmap: enmap.ndmap = None
    lwcs = None
    maps: list[enmap.ndmap] = None
    kmaps: dict[str, list[enmap.ndmap]] = None
    pow_maps: dict[str, enmap.ndmap] = None
    pow_auto_maps: dict[str, enmap.ndmap] = None
    pow_noise_maps: dict[str, enmap.ndmap] = None
    area: float = None
    trimmed = None

    def __init__(
        self,
        maps: list[enmap.ndmap],
        windows: list[enmap.ndmap] = None,
        normalize: str = "phys",
        ell_index: float = None,
    ):
        self.maps = maps
        if windows is not None:
            self.maps = [maps * win for (maps, win) in zip(maps, windows)]
        self.windows = windows
        self.get_ellmaps()
        self.get_kmaps(normalize=normalize)
        self.get_2d_spectra(ell_index=ell_index)

    def copy(self):
        return deepcopy(self)

    def clean_attributes(self, attr_list: list[str]):
        """
        Delete attributes from the instance based on the provided list of names.
        Used to save memory if needed
        """

        for attr in attr_list:
            if hasattr(self, attr):
                delattr(self, attr)

    def get_ellmaps(self):
        """
        Compute useful stuff (modlmap, thetamap etc.)
        """
        enmap_template = self.maps[0]  # assume all map should have same geometry
        self.shape = (enmap_template.shape[-2], enmap_template.shape[-1])
        self.ncomp = 3 if enmap_template.shape[0] == 3 else 1
        self.Nsplits = len(self.maps)
        self.wcs = enmap_template.wcs
        self.lmap = enmap_template.laxes(broadcastable=True)
        self.ly, self.lx = enmap_template.laxes()
        self.lymap, self.lxmap = enmap_template.lmap()
        self.modlmap = enmap_template.modlmap().astype(np.float32)
        self.lwcs = enmap.lwcs(self.shape, self.wcs)
        self.llims = (min(self.lx), max(self.lx), min(self.ly), max(self.ly))
        self.thetamap = np.rad2deg(np.arctan2(self.lymap, self.lxmap))
        # self.xy0, self.xy1 = self.wcs.all_pix2world(0, 0, 0), self.wcs.all_pix2world(
        #     self.shape[0] - 1, self.shape[1] - 1, 0
        # )
        # if self.xy0[0] > self.xy1[0]:
        #     print('sup')
        #     self.pixscale_x = (
        #         np.abs(self.xy1[0] - self.xy0[0])
        #         / self.shape[0]
        #         * np.pi
        #         / 180.0
        #         * np.cos(np.pi / 180.0 * 0.5 * (self.xy0[1] + self.xy1[1]))
        #     )
        # else:
        #     print('inf')
        #     self.pixscale_x = (
        #         np.abs((360.0 - self.xy1[0]) + self.xy0[0])
        #         / self.shape[0]
        #         * np.pi
        #         / 180.0
        #         * np.cos(np.pi / 180.0 * 0.5 * (self.xy0[1] + self.xy1[1]))
        #     )
        # self.pixscale_y = np.abs(self.xy1[1] - self.xy0[1]) / self.shape[1] * np.pi / 180.0
        self.pixscale_x, self.pixscale_y = self.wcs.wcs.cdelt
        self.pixscale_x *= np.pi / 180
        self.pixscale_y *= np.pi / 180
        self.area = (
            self.shape[0]
            * self.shape[1]
            * abs(self.pixscale_x)
            * abs(self.pixscale_y)
            # * (180 / np.pi) ** 2
        )
        self.fsky = self.area / (4 * np.pi)
        # self.fsky = get_fsky(window=self.windows[0])
        # self.area = self.fsky * (4 * np.pi)

    def trim_at_ell(self, ell_trim):
        """
        Trim all existing Fourier maps (ellmaps, kmaps, power maps) at ell trim
        """
        assert ell_trim > 0

        # Trim dict[str, list[enmap.ndmap]]
        for attr_name in ["kmaps"]:
            maps_dict = getattr(self, attr_name)
            if maps_dict is not None:
                maps_dict_trim = {
                    k: [
                        trim_enmap_at_ell(self.lx, self.ly, m, ell_trim=ell_trim)
                        for m in v
                    ]  # FIXME wcs is wrong here
                    for k, v in maps_dict.items()
                }
                setattr(self, attr_name, maps_dict_trim)

        # Trim dict[str, enmap.ndmap]
        for attr_name in ["pow", "pow_noise", "pow_auto"]:
            maps_dict = getattr(self, attr_name)
            if maps_dict is not None:
                maps_dict_trim = {
                    k: trim_enmap_at_ell(
                        self.lx, self.ly, m, ell_trim=ell_trim
                    )  # FIXME wcs is wrong here
                    for k, m in maps_dict.items()
                }
                setattr(self, attr_name, maps_dict_trim)

        # Trim enmap.ndmap
        for attr_name in ["pow_win"]:
            maps = getattr(self, attr_name)
            if maps is not None:
                maps_trim = trim_enmap_at_ell(
                    self.lx, self.ly, maps, ell_trim=ell_trim
                )  # FIXME wcs is wrong here
                setattr(self, attr_name, maps_trim)

        self.trimmed = ell_trim
        idx = (self.lx < ell_trim) & (self.lx > -ell_trim)
        idy = (self.ly < ell_trim) & (self.ly > -ell_trim)
        mask_2d = np.ix_(idy, idx)
        # self.lmap = self.lmap[idy, idx] # TODO : mask lmap
        self.ly, self.lx = self.ly[idy], self.lx[idx]
        self.lymap, self.lxmap = self.lymap[mask_2d], self.lxmap[mask_2d]
        self.modlmap = self.modlmap[mask_2d]
        self.llims = (min(self.lx), max(self.lx), min(self.ly), max(self.ly))
        self.thetamap = np.rad2deg(np.arctan2(self.lymap, self.lxmap))

    def get_kmaps(self, normalize="phys"):

        # Compute TQU kmaps and map them into self.kmap dict
        kmaps_pixell = [
            enmap.fft(enmap_, normalize="phys")
            for enmap_ in self.maps
        ]
        self.kmaps = {
            TQU: [kmap[TQU_indices[TQU]] for kmap in kmaps_pixell] for TQU in "TQU"
        }

        # Compute EB kmaps
        cos2theta = np.cos(2 * np.deg2rad(self.thetamap))
        sin2theta = np.sin(2 * np.deg2rad(self.thetamap))
        self.kmaps["E"] = [
            -self.kmaps["Q"][i] * cos2theta - self.kmaps["U"][i] * sin2theta
            for i in range(self.Nsplits)
        ]
        self.kmaps["B"] = [
            self.kmaps["Q"][i] * sin2theta - self.kmaps["U"][i] * cos2theta
            for i in range(self.Nsplits)
        ]

    def get_win_spec(self, normalize="phys", ell_index=None):
        # kmap_win = np.fft.fft2(
        kmap_win = (
            enmap.fft(self.windows[0], normalize="phys")
        )  # TODO: make this work for all windows
        self.pow_win = (kmap_win * np.conj(kmap_win)).real
        fac = 1.0 if ell_index is None else (self.modlmap**ell_index / (2 * np.pi))
        fac *= (self.area / (4 * np.pi))
        self.pow_win *= fac

    def get_pixel_window(self):  # TODO: problem with trimmed ?
        pixW = np.sinc(self.lx[self.ix] * self.pixScaleX / (2.0 * np.pi)) * np.sinc(
            self.ly[self.iy] * self.pixScaleY / (2.0 * np.pi)
        )
        pixW = pixW**2

    def get_2d_spectra(self, ell_index=None, skip_useless=True):
        self.pow_crosses: dict[str, list[enmap.ndmap]] = (
            {}
        )  # stores all individual spectra
        self.pow_autos: dict[str, list[enmap.ndmap]] = {}  # stores all indivual auto
        self.pow: dict[str, enmap.ndmap] = {}  # mean of cross spectra
        self.pow_auto: dict[str, enmap.ndmap] = {}  # mean of auto spectra
        self.pow_noise: dict[str, enmap.ndmap] = {}  # auto - cross
        for X, Y in itertools.product("TQUEB", repeat=2):
            if skip_useless & (
                ((X in "QU") & (Y in "EB")) | ((X in "EB") & (Y in "QU"))
            ):
                continue
            self.pow_crosses[X + Y] = []
            self.pow_autos[X + Y] = []
            self.pow[X + Y] = enmap.zeros(self.shape, self.lwcs, dtype=np.complex128)
            self.pow_auto[X + Y] = enmap.zeros(
                self.shape, self.lwcs, dtype=np.complex128
            )
            for i1, i2 in itertools.combinations_with_replacement(
                range(self.Nsplits), r=2
            ):
                pow_iter = (self.kmaps[Y][i2] * np.conj(self.kmaps[X][i1])).real
                if i1 != i2:
                    self.pow_crosses[X + Y].append(pow_iter)
                    self.pow[X + Y] += pow_iter
                elif i1 == i2:
                    self.pow_autos[X + Y].append(pow_iter)
                    self.pow_auto[X + Y] += pow_iter

            # Divide by the number of iterations since we add all iterations
            self.pow[X + Y] /= self.Nsplits * (self.Nsplits - 1) / 2
            self.pow_auto[X + Y] /= self.Nsplits
            self.pow_noise[X + Y] = (
                self.pow_auto[X + Y] - self.pow[X + Y]
            ) / self.Nsplits

            fac = 1.0 if ell_index is None else self.modlmap**ell_index / (2 * np.pi)
            fac *= (self.area / (4 * np.pi))
            self.pow[X + Y] *= fac
            self.pow_auto[X + Y] *= fac
            self.pow_noise[X + Y] *= fac

    def axplot(
        self,
        map_to_plot=None,
        comp=None,  # 'TT', 'TE', ... if map is a dict
        colorbar=False,
        ell_index=2,
        log=False,
        zoom=1000,
        ax_to_plot=None,
        downgrade=2,
        **args,
    ):
        """Plots self.pow by default.
        Also return the imshow() mappable (for colorbar etc.)
        """

        map_to_plot = self.pow.copy() if map_to_plot is None else map_to_plot.copy()

        if type(map_to_plot) == dict:
            map_to_plot = map_to_plot[comp].copy()

        if ell_index != 0:
            map_to_plot *= self.modlmap ** (2)

        if downgrade != 1:
            map_to_plot = map_to_plot.downgrade(downgrade)

        if log:
            map_to_plot = np.log10(map_to_plot.astype(np.float64))

        plot = ax_to_plot.imshow(
            np.fft.fftshift(map_to_plot.real.astype(np.float64)),
            extent=self.llims,
            **args,
        )

        if colorbar:
            plt.colorbar(plot, location="bottom")

        ax_to_plot.set_xlim(-zoom, zoom)
        ax_to_plot.set_ylim(-zoom, zoom)
        ax_to_plot.set_xlabel(r"$\ell_x$")
        ax_to_plot.set_ylabel(r"$\ell_y$")
        return plot

    def plot(self, **args):
        """
        Plots the pow map using enplot, much slower than axplot()
        """
        enplot.pshow(enmap.enmap(np.fft.fftshift(self.pow), self.lwcs), **args)

    def radial_binned_map(
        self,
        bin_edges: np.ndarray,
        which_map: dict[str, enmap.ndmap] = None,
        TQU: str = None,
    ) -> enmap.ndmap:
        """Bins a given map (self.pow by default).

        Args:
            bin_edges (_type_): ell-indices for separation between bins
            which_map (_type_, optional): Must have same pixellization as self.pow. If None uses self.pow.
            TQU (_type_, optional): If ncomp=3, choose which one to use between to I, Q and U. If None uses I.

        Returns:
            _type_: _description_
        """
        which_map = self.pow if which_map is None else which_map

        rad_binned_map: dict[str, enmap.ndmap] = {}
        for comp, spec_map in which_map.items():
            # Define a mask in kspace using theta map
            smap_flatten = spec_map.flatten()

            # Create a map of where bins are
            bin_map = np.digitize(self.modlmap, bin_edges, right=True)
            bin_map_flatten = bin_map.flatten()

            # Bin the map and divide by the occupation number
            rbin_map = np.bincount(bin_map_flatten, weights=smap_flatten)
            bincount = np.bincount(bin_map_flatten)
            rad_binned_map[comp] = (rbin_map / bincount)[bin_map]
        return rad_binned_map

    def radial_binned_1d_spec(
        self, bin_edges, which_map=None, theta_range=None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:

        if type(which_map) != dict and which_map is not None:
            which_map = {"I": which_map}
        smap_dict: dict[str, enmap.ndmap] = self.pow if which_map is None else which_map

        # Define a mask in kspace using theta map
        theta_range = [0, 360] if theta_range is None else theta_range
        theta_mask = np.where(
            (theta_range[0] <= self.thetamap.copy().flatten())
            & (theta_range[1] > self.thetamap.copy().flatten())
        )

        rad_bin_spec = {}  # Cls or Dls
        for spec, smap in smap_dict.items():
            smap_flatten = smap.flatten()[theta_mask]

            # Create a map of where bins are
            bin_map = np.digitize(self.modlmap, bin_edges, right=True)
            bin_map_flatten = bin_map.flatten()[theta_mask]

            # Bin the map and divide by the occupation number
            bincount = np.bincount(bin_map_flatten)
            rbin_map = np.bincount(
                bin_map_flatten, weights=smap_flatten.astype(np.float64)
            )
            bins_center = (bin_edges[:-1] + bin_edges[1:]) / 2
            rad_bin_spec[spec] = (
                rbin_map[1:-1] / bincount[1:-1]
            )  # First bin is before the first bin edge and last one is after last bin edge
        return bins_center, rad_bin_spec

    def subtract_rad_profile(self, bin_edges, inplace=False):
        rad_binned_map = self.radial_binned_map(bin_edges)
        pow_subtracted = self.pow - rad_binned_map
        if not inplace:
            return pow_subtracted
        else:
            self.pow = pow_subtracted

    def write_so_kspec_pickle(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def make_1d_spectra_and_save(kspec: So_Spec2D, bin_edges, theta_ranges, filename):
    ls = (bin_edges[1:] + bin_edges[:-1]) / 2
    # Start with saving 1d radial power spectra and noise spectra
    ps_full = {}
    ps_full_noise = {}
    for comp in ["T", "Q", "U"]:
        ps_full[comp] = kspec.radial_binned_1d_spec(bin_edges=bin_edges, TQU=comp)[1]
        ps_full_noise[comp] = kspec.radial_binned_1d_spec(
            bin_edges=bin_edges, which_map=kspec.pow_noise, TQU=comp
        )[1]

    # Make 1d radial power and noise spectra for given theta_ranges
    ps_thetas = {}
    ps_thetas_noise = {}
    for t, theta_range in enumerate(theta_ranges):
        range_name = f"theta_{t}"
        ps_thetas[range_name] = {}
        ps_thetas_noise[range_name] = {}
        for comp in ["T", "Q", "U"]:
            ps_thetas[range_name][comp] = kspec.radial_binned_1d_spec(
                bin_edges=bin_edges, TQU=comp, theta_range=theta_range
            )[1]
            ps_thetas_noise[range_name][comp] = kspec.radial_binned_1d_spec(
                bin_edges=bin_edges,
                which_map=kspec.pow_noise,
                TQU=comp,
                theta_range=theta_range,
            )[1]

    # Put everything in a dict and save it
    save_dict = {
        "theta_range": theta_ranges,
        "ls": ls,
        "ps_full": ps_full,
        "ps_thetas": ps_thetas,
        "ps_full_noise": ps_full_noise,
        "ps_thetas_noise": ps_thetas_noise,
    }

    with open(filename, "wb") as f:
        pickle.dump(save_dict, f)


def read_so_kspec_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


# def from_enmap(enmap_: enmap.ndmap) -> So_Spec2D:
#     kspec = So_Spec2D()

#     # some geometry and misc
#     kspec.shape = (enmap_.shape[-2], enmap_.shape[-1])
#     kspec.ncomp = 3 if enmap_.shape[0] == 3 else 1
#     kspec.Nsplits = 1
#     kspec.wcs = enmap_.wcs
#     kspec.ly, kspec.lx = enmap_.laxes()
#     kspec.lymap, kspec.lxmap = enmap_.lmap()
#     kspec.modlmap = enmap_.modlmap().astype(np.float32)
#     kspec.lwcs = enmap.lwcs(enmap_.shape, enmap_.wcs)
#     kspec.llims = (min(kspec.lx), max(kspec.lx), min(kspec.ly), max(kspec.ly))
#     kspec.thetamap = np.rad2deg(np.arctan2(kspec.lymap, kspec.lxmap))

#     # compute kmaps and kspecs
#     kmap = enmap.fft(enmap_)
#     kspec.kmaps = [kmap]
#     kspec.pow = (kmap * np.conj(kmap)).real
#     kspec.pow *= kspec.modlmap**2  # D_ell because why not

#     return kspec


# def from_enmap_list(enmap_list: list[enmap.ndmap]) -> So_Spec2D:
#     N_splits = len(enmap_list)

#     kspec = So_Spec2D()
#     enmap_template = enmap_list[0]  # assume all map should have same geometry
#     kspec.shape = (enmap_template.shape[-2], enmap_template.shape[-1])
#     kspec.ncomp = 3 if enmap_template.shape[0] == 3 else 1
#     kspec.Nsplits = N_splits
#     kspec.wcs = enmap_template.wcs
#     kspec.lmap = enmap_template.laxes(broadcastable=True)
#     kspec.ly, kspec.lx = enmap_template.laxes()
#     kspec.lymap, kspec.lxmap = enmap_template.lmap()
#     kspec.modlmap = enmap_template.modlmap().astype(np.float32)
#     kspec.lwcs = enmap.lwcs(kspec.shape, kspec.wcs)
#     kspec.llims = (min(kspec.lx), max(kspec.lx), min(kspec.ly), max(kspec.ly))
#     kspec.thetamap = np.rad2deg(np.arctan2(kspec.lymap, kspec.lxmap))

#     # start with kmaps
#     kspec.kmaps = [enmap.fft(enmap_) for enmap_ in enmap_list]

#     kspec.kmaps_dict = {
#         "T": [kmap[0] for kmap in kspec.kmaps],
#         "Q": [kmap[1] for kmap in kspec.kmaps],
#         "U": [kmap[2] for kmap in kspec.kmaps],
#     }

#     rot = enmap.qued_rotmat(kspec.lmap, spin=2)
#     kmaps_EB = enmap.matmul(rot, kspec.kmaps[..., 1:2])
#     # combine kmaps for kspecs cross and autos
#     kspec.pow_crosses = []
#     kspec.pow_autos = []
#     kspec.pow = enmap.zeros(shape=enmap_template.shape, wcs=kspec.lwcs)
#     kspec.pow_auto = enmap.zeros(shape=enmap_template.shape, wcs=kspec.lwcs)
#     for i1, i2 in itertools.combinations_with_replacement(range(N_splits), r=2):
#         assert test_same_geometry(
#             enmap_list[i1], enmap_list[i2]
#         ), "All maps must have same geometry"

#         # if kspec.ncomp==1:
#         #     pow_iter = (kspec.kmaps[i1] * np.conj(kspec.kmaps[i2])).real * kspec.modlmap**2
#         # elif kspec.ncomp==3:
#         #     pow_iter = enmap.zeros(shape=kspec.kmaps[0].shape, wcs=kspec.kmaps[0].wcs)
#         #     for i in range(3):
#         #         pow_iter[i] = (kspec.kmaps[i1][i] * np.conj(kspec.kmaps[i2][i])).real * kspec.modlmap**2
#         pow_iter = (kspec.kmaps[i1] * np.conj(kspec.kmaps[i2])).real * kspec.modlmap**2

#         if i1 != i2:
#             kspec.pow_crosses.append(pow_iter)
#             kspec.pow += pow_iter
#         elif i1 == i2:
#             kspec.pow_autos.append(pow_iter)
#             kspec.pow_auto += pow_iter

#     # Divide by the number of iterations since we add all iterations
#     kspec.pow /= N_splits * (N_splits - 1) / 2
#     kspec.pow_auto /= N_splits
#     kspec.pow_noise = (kspec.pow_auto - kspec.pow) / N_splits

#     kspec.pow_maps = {}
#     kspec.pow_auto_maps = {}
#     kspec.pow_noise_maps = {}

#     # Create E and B kmaps from Q and U
#     kspec.kmaps_EB = [
#         enmap.zeros((2, *kspec.shape), kmap.wcs, dtype=np.complex128)
#         for kmap in kspec.kmaps
#     ]
#     for i in range(len(kspec.kmaps_EB)):
#         kspec.kmaps_EB[i][0] = kspec.kmaps[i][1] * np.cos(
#             2 * np.deg2rad(kspec.thetamap)
#         ) + kspec.kmaps[i][2] * np.sin(2 * np.deg2rad(kspec.thetamap))
#         kspec.kmaps_EB[i][1] = kspec.kmaps[i][1] * np.sin(
#             2 * np.deg2rad(kspec.thetamap)
#         ) - kspec.kmaps[i][2] * np.cos(2 * np.deg2rad(kspec.thetamap))

#     # combine kmaps for kspecs cross and autos
#     kspec.pow_EB_crosses = []
#     kspec.pow_EB_autos = []
#     kspec.pow_EB = enmap.zeros(shape=(2, *kspec.shape), wcs=kspec.lwcs)
#     kspec.pow_EB_auto = enmap.zeros(shape=(2, *kspec.shape), wcs=kspec.lwcs)
#     for i1, i2 in itertools.combinations_with_replacement(range(kspec.Nsplits), r=2):

#         pow_iter = (
#             kspec.kmaps_EB[i1] * np.conj(kspec.kmaps_EB[i2])
#         ).real * kspec.modlmap**2

#         if i1 != i2:
#             kspec.pow_EB_crosses.append(pow_iter)
#             kspec.pow_EB += pow_iter
#         elif i1 == i2:
#             kspec.pow_EB_autos.append(pow_iter)
#             kspec.pow_EB_auto += pow_iter

#     # Divide by the number of iterations since we add all iterations
#     kspec.pow_EB /= kspec.Nsplits * (kspec.Nsplits - 1) / 2
#     kspec.pow_EB_auto /= kspec.Nsplits
#     kspec.pow_EB_noise = (kspec.pow_EB_auto - kspec.pow_EB) / kspec.Nsplits

#     return kspec
