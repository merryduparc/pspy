"""
Defines a "kspec" class for making 2-D angular power spectra of CMB maps (CAR pixellization)
"""

from pixell import enmap, enplot
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import pickle
import fortran
import itertools
from astropy import wcs
from tqdm import tqdm
from pspy.so_window import get_survey_solid_angle_ndmap, get_fsky_ndmap
from pspy.pspy_utils import read_binning_file

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
        self.patch_fsky = get_fsky_ndmap(self.windows[0] * 0. + 1.)
        self.patch_area = get_survey_solid_angle_ndmap(self.windows[0] * 0. + 1.)
        self.fsky = get_fsky_ndmap(self.windows[0])
        self.area = get_survey_solid_angle_ndmap(self.windows[0])

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

    def get_kmaps(self):
        """
        Computes k-space maps from enmaps.
        """

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

    def get_win_spec(self, ell_index=None):
        """
        Computes the window power map. Only uses first input window for now.
        """
        kmap_win = (
            enmap.fft(self.windows[0], normalize="phys")
        )  # TODO: make this work for different windows
        self.pow_win = (kmap_win * np.conj(kmap_win)).real
        fac = 1.0 if ell_index is None else (self.modlmap**ell_index)
        fac *= (self.patch_area / (4 * np.pi))
        self.pow_win *= fac
        
    # TODO make this one work
    # def get_pixel_window(self): 
    #     pixW = np.sinc(self.lx[self.ix] * self.pixScaleX / (2.0 * np.pi)) * np.sinc(
    #         self.ly[self.iy] * self.pixScaleY / (2.0 * np.pi)
    #     )
    #     pixW = pixW**2

    def get_2d_spectra(self, ell_index=None, skip_useless=True):
        """
        From kmaps, computes all power maps from kmaps combinations.
        Then makes mean of the cross and auto, and a noise power map.
        """
        self.pow_crosses: dict[str, list[enmap.ndmap]] = (
            {}
        )  # stores all individual cross
        self.pow_autos: dict[str, list[enmap.ndmap]] = {}  # stores all indivual auto
        self.pow: dict[str, enmap.ndmap] = {}  # mean of cross spectra
        self.pow_auto: dict[str, enmap.ndmap] = {}  # mean of auto spectra
        self.pow_noise: dict[str, enmap.ndmap] = {}  # mean auto - mean cross
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

            # Divide by the number of iterations to obtain the mean
            self.pow[X + Y] /= self.Nsplits * (self.Nsplits - 1) / 2
            self.pow_auto[X + Y] /= self.Nsplits
            self.pow_noise[X + Y] = (
                self.pow_auto[X + Y] - self.pow[X + Y]
            ) / self.Nsplits

            fac = 1.0 if ell_index is None else self.modlmap**ell_index
            fac *= (self.patch_area / (4 * np.pi))
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
        """Plots a given map, self.pow by default. **args goes into imshow.
        Also return the imshow() mappable (for colorbar etc.)
        """

        map_to_plot = self.pow.copy() if map_to_plot is None else map_to_plot.copy()

        if type(map_to_plot) == dict:
            map_to_plot = map_to_plot[comp].copy()

        if ell_index is not None:
            map_to_plot *= self.modlmap ** ell_index

        if downgrade != 1 and downgrade is not None:
            map_to_plot = map_to_plot.downgrade(downgrade)

        if log:
            map_to_plot = np.log10(map_to_plot.astype(np.float64))

        plot = ax_to_plot.imshow(
            np.fft.fftshift(map_to_plot.real.astype(np.float64)),
            extent=self.llims,
            origin='lower',
            **args,
        )

        if colorbar:
            plt.colorbar(plot, location="bottom")

        ax_to_plot.set_xlim(-zoom, zoom)
        ax_to_plot.set_ylim(-zoom, zoom)
        ax_to_plot.set_xlabel(r"$\ell_x$")
        ax_to_plot.set_ylabel(r"$\ell_y$")
        return plot

    def radial_binned_map(      # TODO : change bin_edges for a binning_file
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

    def radial_binned_1d_spec(      # TODO : change bin_edges for a binning_file
        self, bin_edges, which_map=None, theta_range=None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        makes a 1D power spectra from power map
        """

        if type(which_map) != dict and which_map is not None:
            which_map = {"I": which_map}
        smap_dict: dict[str, enmap.ndmap] = self.pow if which_map is None else which_map

        # Define a mask in kspace using theta map
        theta_range = [0, 180] if theta_range is None else theta_range
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
        """
        Subtract the radial profile of a map (using self.radial_binned_map)
        """
        rad_binned_map = self.radial_binned_map(bin_edges)
        pow_subtracted = self.pow - rad_binned_map
        if not inplace:
            return pow_subtracted
        else:
            self.pow = pow_subtracted

    def write_so_kspec_pickle(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def make_1d_spectra_and_save(spec2d: So_Spec2D, bin_edges, theta_ranges, filename):
    # TODO: change this so it save with so_spectra.write_ps instead
    ls = (bin_edges[1:] + bin_edges[:-1]) / 2
    # Start with saving 1d radial power spectra and noise spectra
    ps_full = {}
    ps_full_noise = {}
    for comp in ["T", "Q", "U"]:
        ps_full[comp] = spec2d.radial_binned_1d_spec(bin_edges=bin_edges, TQU=comp)[1]
        ps_full_noise[comp] = spec2d.radial_binned_1d_spec(
            bin_edges=bin_edges, which_map=spec2d.pow_noise, TQU=comp
        )[1]

    # Make 1d radial power and noise spectra for given theta_ranges
    ps_thetas = {}
    ps_thetas_noise = {}
    for t, theta_range in enumerate(theta_ranges):
        range_name = f"theta_{t}"
        ps_thetas[range_name] = {}
        ps_thetas_noise[range_name] = {}
        for comp in ["T", "Q", "U"]:
            ps_thetas[range_name][comp] = spec2d.radial_binned_1d_spec(
                bin_edges=bin_edges, TQU=comp, theta_range=theta_range
            )[1]
            ps_thetas_noise[range_name][comp] = spec2d.radial_binned_1d_spec(
                bin_edges=bin_edges,
                which_map=spec2d.pow_noise,
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

def read_so_2dspec_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def get_powsum_maps(spec2d:So_Spec2D, binning_file: str, lmax: float) -> dict[np.ndarray]:
    """Computes the sums of binning maps times window spec using a fortran routine
    """
    bin_low, bin_high, bin_cent, bin_size = read_binning_file(binning_file, lmax=lmax, start_at_two=False)

    spec2d_wide = spec2d.copy()
    spec2d_wide.trim_at_ell(2 * lmax + 500)

    pow_shifted = np.fft.fftshift(spec2d_wide.pow_win.astype(np.float64))
    lx_shifted = np.fft.fftshift(spec2d_wide.lx.copy())
    ly_shifted = np.fft.fftshift(spec2d_wide.ly.copy())

    lx = spec2d_wide.lx.copy()
    ly = spec2d_wide.ly.copy()

    modlmap = spec2d_wide.modlmap.copy()
    
    spec2d_trim = spec2d.copy()
    spec2d_trim.trim_at_ell(lmax)

    ang=np.fft.fftshift(np.arctan2(spec2d_trim.lymap,spec2d_trim.lxmap))
    cos_array=np.cos(-2*ang)
    sin_array=np.sin(-2*ang)

    pMMaps_0 = np.zeros((len(bin_low), len(spec2d_trim.ly), len(spec2d_trim.lx)), dtype=np.float64)
    pMMaps_cos = pMMaps_0.copy()
    pMMaps_sin =  pMMaps_0.copy()
    pMMaps_cos2 =  pMMaps_0.copy()
    pMMaps_sin2 =  pMMaps_0.copy()
    pMMaps_cossin =  pMMaps_0.copy()

    for ibin in tqdm(range(len(bin_low)), desc='Looping over bins ', smoothing=1.):

        location = np.where((modlmap >= bin_low[ibin]) & (modlmap <= bin_high[ibin]))

        binMap = pow_shifted.copy() * 0.
        binMap[location] = 1. # shifed
        sumBin = binMap.sum()
        
        binMap0 = spec2d_trim.pow_win.copy() * 0.
        binMap_cos = binMap0.copy()
        binMap_sin = binMap0.copy()
        binMap_cos2 = binMap0.copy()
        binMap_sin2 = binMap0.copy()
        binMap_cossin = binMap0.copy()
        
        fortran.fortran(
                iy=location[0],  # no shift, no trim
                ix=location[1],  # no shift, no trim
                bin_ly=ly,  # no shift, no trim
                bin_lx=lx,  # no shift, no trim
                pmshift_11=pow_shifted.T,  # shifted, no trim
                pmshift_12=pow_shifted.T,  # shifted, no trim
                pmshift_22=pow_shifted.T,  # shifted, no trim
                cos_array=cos_array.T,
                sin_array=sin_array.T,
                bmap=None,
                bmap0=binMap0.T,  # shifted, trimmed
                bmap_cos=binMap_cos.T,
                bmap_sin=binMap_sin.T,
                bmap_cos2=binMap_cos2.T,
                bmap_sin2=binMap_sin2.T,
                bmap_cossin=binMap_cossin.T,
                phlx=lx_shifted,  # shifted, no trim
                phly=ly_shifted,  # shifted, no trim
                type_bn=0,
                trimatl=lmax,
            )
        
        pMMaps_0[ibin] = binMap0 / sumBin
        pMMaps_cos[ibin] = binMap_cos/sumBin
        pMMaps_sin[ibin] = binMap_sin/sumBin
        pMMaps_cos2[ibin] = binMap_cos2/sumBin
        pMMaps_sin2[ibin] = binMap_sin2/sumBin
        pMMaps_cossin[ibin] = binMap_cossin/sumBin

    powsum_maps = {
        '0': pMMaps_0,
        'cos': pMMaps_cos,
        'sin': pMMaps_sin,
        'cos2': pMMaps_cos2,
        'sin2': pMMaps_sin2,
        'cossin': pMMaps_cossin,
    }
    return powsum_maps


def get_trig_M_arrays(powsum_maps:dict[np.ndarray], spec2d:So_Spec2D, binning_file: str, lmax: float) -> np.ndarray:
    
    bin_low, bin_high, bin_cent, bin_size = read_binning_file(binning_file, lmax=lmax, start_at_two=False)

    spec2d_trim = spec2d.copy()
    spec2d_trim.trim_at_ell(lmax)
    
    mArray_zeros = np.zeros(shape=(len(bin_low),len(bin_low)))

    mArrays = {
        trig: mArray_zeros.copy() for trig in powsum_maps.keys()
    }

    for j in range(len(bin_low)):
        modlmap_trimmed = spec2d_trim.modlmap
        location = np.where(
            (modlmap_trimmed >= bin_low[j]) &\
            (modlmap_trimmed <= bin_high[j])
        )
        binMapTrim = spec2d_trim.pow_win.copy()*0.
        binMapTrim[location] = 1.
        # binMapTrim[location] *= np.nan_to_num(1./(powerMaskHolderTrim.modlmap[location])**powerOfL)
        for trig in powsum_maps.keys():
            for i in range(len(powsum_maps[trig])):
                newMap = powsum_maps[trig][i].copy()
                result = (newMap * np.fft.ifftshift(binMapTrim)).sum()
                mArrays[trig][i, j] = result
    
    return mArrays

def get_mode_coupling_2D(mArrays, spec2d) -> np.ndarray:

    n = len(mArrays['0'])

    mcm = np.zeros((6*n, 6*n), dtype=mArrays['0'].dtype)

    mcm[0*n:1*n, 0*n:1*n] = mArrays['0']

    mcm[1*n:2*n, 1*n:2*n] = mArrays['cos']
    mcm[1*n:2*n, 2*n:3*n] = -mArrays['sin']
    mcm[2*n:3*n, 1*n:2*n] = mArrays['sin']
    mcm[2*n:3*n, 2*n:3*n] = mArrays['cos']

    mcm[3*n:4*n, 3*n:4*n] = mArrays['cos2']
    mcm[3*n:4*n, 4*n:5*n] = -2 * mArrays['cossin']
    mcm[3*n:4*n, 5*n:6*n] = mArrays['sin2']
    mcm[4*n:5*n, 3*n:4*n] = mArrays['cossin']
    mcm[4*n:5*n, 4*n:5*n] = mArrays['cos2'] - mArrays['sin2']
    mcm[4*n:5*n, 5*n:6*n] = -mArrays['cossin']
    mcm[5*n:6*n, 3*n:4*n] = mArrays['sin2']
    mcm[5*n:6*n, 4*n:5*n] = 2 * mArrays['cossin']
    mcm[5*n:6*n, 5*n:6*n] = mArrays['cos2']
    
    return mcm / spec2d.patch_area