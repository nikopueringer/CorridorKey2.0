"""Shared analytic camera log/gamut transforms.

Camera curves use the pinned colour-science implementation where available;
Samsung Log and RED Log3G10 use their published equations directly. Gamut
conversion is linear RGB -> XYZ D65 -> linear Rec.709, with no clipping, LUT,
tone mapping, or chromatic adaptation.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    id: str
    label: str
    gamut: str
    curve: str


PROFILES: dict[str, CameraProfile] = {
    "legacy": CameraProfile("legacy", "Legacy / standard video", "ITU-R BT.709", "legacy"),
    "sony_slog3_sgamut3": CameraProfile("sony_slog3_sgamut3", "Sony S-Log3 · S-Gamut3", "Sony S-Gamut3", "slog3"),
    "sony_slog3_sgamut3_cine": CameraProfile("sony_slog3_sgamut3_cine", "Sony S-Log3 · S-Gamut3.Cine", "Sony S-Gamut3.Cine", "slog3"),
    "red_log3g10_redwidegamut": CameraProfile("red_log3g10_redwidegamut", "RED Log3G10 · REDWideGamutRGB", "REDWideGamutRGB", "red_log3g10"),
    "blackmagic_film_gen5_bmdwidegamut": CameraProfile("blackmagic_film_gen5_bmdwidegamut", "Blackmagic Film Gen 5 · Wide Gamut Gen 5", "Blackmagic Wide Gamut", "bmd_gen5"),
    "canon_log2_cinema_gamut": CameraProfile("canon_log2_cinema_gamut", "Canon Log 2 · Cinema Gamut", "Canon Cinema Gamut", "canon_log2"),
    "canon_log3_cinema_gamut": CameraProfile("canon_log3_cinema_gamut", "Canon Log 3 · Cinema Gamut", "Canon Cinema Gamut", "canon_log3"),
    "fuji_flog_f_gamut": CameraProfile("fuji_flog_f_gamut", "Fujifilm F-Log · F-Gamut", "F-Gamut", "flog"),
    "fuji_flog2_f_gamut": CameraProfile("fuji_flog2_f_gamut", "Fujifilm F-Log2 · F-Gamut", "F-Gamut", "flog2"),
    "fuji_flog2c_f_gamut_c": CameraProfile("fuji_flog2c_f_gamut_c", "Fujifilm F-Log2C · F-Gamut C", "F-Gamut C", "flog2"),
    "panasonic_vlog_v_gamut": CameraProfile("panasonic_vlog_v_gamut", "Panasonic V-Log · V-Gamut", "Panasonic V-Gamut", "vlog"),
    "dji_dlog_dgamut": CameraProfile("dji_dlog_dgamut", "DJI D-Log · D-Gamut", "DJI D-Gamut", "dji_dlog"),
    "arri_logc3_awg3_ei800": CameraProfile("arri_logc3_awg3_ei800", "ARRI LogC3 (EI 800) · AWG3", "ARRI Wide Gamut 3", "arri_logc3"),
    "arri_logc4_awg4": CameraProfile("arri_logc4_awg4", "ARRI LogC4 · AWG4", "ARRI Wide Gamut 4", "arri_logc4"),
    "apple_log_rec2020": CameraProfile("apple_log_rec2020", "Apple Log · Rec.2020", "ITU-R BT.2020", "apple_log"),
    "apple_log2_awg2": CameraProfile("apple_log2_awg2", "Apple Log 2 · Apple Wide Gamut", "Apple Wide Gamut 2", "apple_log"),
    "samsung_log_bt2020": CameraProfile("samsung_log_bt2020", "Samsung Log · BT.2020", "ITU-R BT.2020", "samsung_log"),
}


def _colour():
    # colour-science is imported only when a transform is requested, avoiding
    # its Qt startup cost and optional matplotlib warning in normal UI launch.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Matplotlib.*")
        import colour
    return colour


_PRIMARIES = {
    "Sony S-Gamut3": ((.730, .280), (.140, .855), (.100, -.050)),
    "Sony S-Gamut3.Cine": ((.766, .275), (.225, .800), (.089, -.087)),
    "Canon Cinema Gamut": ((.740, .270), (.170, 1.140), (.080, -.100)),
    "F-Gamut C": ((.73470, .26530), (.02630, .97370), (.11730, -.02240)),
    "Panasonic V-Gamut": ((.730, .280), (.165, .840), (.100, -.030)),
    "Apple Wide Gamut 2": ((.725, .301), (.221, .814), (.068, -.076)),
}
_WHITE_D65 = (0.3127, 0.3290)
_MATRIX_CACHE: dict[str, np.ndarray] = {}


def _rgb_to_xyz(gamut: str) -> np.ndarray:
    cached = _MATRIX_CACHE.get(gamut)
    if cached is not None:
        return cached
    colour = _colour()
    if gamut in _PRIMARIES:
        matrix = colour.normalised_primary_matrix(
            np.asarray(_PRIMARIES[gamut], dtype=np.float64), np.asarray(_WHITE_D65, dtype=np.float64))
    else:
        try:
            matrix = colour.RGB_COLOURSPACES[gamut].matrix_RGB_to_XYZ
        except KeyError as exc:
            raise ValueError(f"unsupported RGB gamut {gamut!r}") from exc
    result = np.asarray(matrix, dtype=np.float32)
    _MATRIX_CACHE[gamut] = result
    return result


def _curve(rgb: np.ndarray, curve: str, encode: bool) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    if curve == "legacy":
        return x.copy()
    if curve == "red_log3g10":
        # RED Log3G10 / REDWideGamutRGB white paper.
        y = np.empty_like(x)
        if encode:
            z = x + np.float32(.01)
            low = z < 0
            y[low] = z[low] * np.float32(15.1927)
            y[~low] = np.float32(.224282) * np.log10(z[~low] * np.float32(155.975327) + 1)
        else:
            low = x < 0
            y[low] = x[low] / np.float32(15.1927) - np.float32(.01)
            y[~low] = (np.power(10., x[~low] / np.float32(.224282)) - 1) / np.float32(155.975327) - np.float32(.01)
        return y
    if curve == "samsung_log":
        y = np.empty_like(x)
        if encode:
            low, middle = x < -.05, (x >= -.05) & (x < .01)
            high = x >= .01
            y[low] = 0
            y[middle] = -.20942 * np.log10(-x[middle] + .016904) - .24597
            y[high] = .258984868 * np.log10(x[high] + .0003645) + .720504856
        else:
            low, toe = x < 0, (x >= 0) & (x < .206561909)
            high = x >= .206561909
            y[low] = -.05
            y[toe] = -np.power(10., (x[toe] + .24597) / -.20942) + .016904
            y[high] = np.power(10., (x[high] - .720504856) / .258984868) - .0003645
        return y

    colour = _colour().models
    funcs = {
        "slog3": (colour.log_encoding_SLog3, colour.log_decoding_SLog3, {}),
        "bmd_gen5": (colour.oetf_BlackmagicFilmGeneration5, colour.oetf_inverse_BlackmagicFilmGeneration5, {}),
        "canon_log2": (colour.log_encoding_CanonLog2, colour.log_decoding_CanonLog2, {}),
        "canon_log3": (colour.log_encoding_CanonLog3, colour.log_decoding_CanonLog3, {}),
        "flog": (colour.log_encoding_FLog, colour.log_decoding_FLog, {}),
        "flog2": (colour.log_encoding_FLog2, colour.log_decoding_FLog2, {}),
        "vlog": (colour.log_encoding_VLog, colour.log_decoding_VLog, {}),
        "dji_dlog": (colour.log_encoding_DJIDLog, colour.log_decoding_DJIDLog, {}),
        "arri_logc3": (colour.log_encoding_ARRILogC3, colour.log_decoding_ARRILogC3,
                       {"firmware": "SUP 3.x", "method": "Linear Scene Exposure Factor", "EI": 800}),
        "arri_logc4": (colour.log_encoding_ARRILogC4, colour.log_decoding_ARRILogC4, {}),
        "apple_log": (colour.log_encoding_AppleLogProfile, colour.log_decoding_AppleLogProfile, {}),
    }
    try:
        func = funcs[curve][0 if encode else 1]
    except KeyError as exc:
        raise ValueError(f"unknown camera transfer curve {curve!r}") from exc
    # Several vectorized reference curves evaluate both sides of a piecewise
    # expression; the unused branch can legitimately contain log-domain NaNs.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.asarray(func(x, **funcs[curve][2]), dtype=np.float32)


def _matrix_transform(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.matmul(np.asarray(rgb, dtype=np.float32), matrix.T).astype(np.float32, copy=False)


def _apply_tiled(image: np.ndarray, transform) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim < 1 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB with 3 channels in the last dimension, got {arr.shape}")
    if arr.ndim < 3:
        return transform(arr)
    out = np.empty_like(arr)
    height_axis = 0 if arr.ndim == 3 else arr.ndim - 3
    for top in range(0, arr.shape[height_axis], 128):
        slices = [slice(None)] * arr.ndim
        slices[height_axis] = slice(top, top + 128)
        selection = tuple(slices)
        out[selection] = transform(arr[selection])
    return out


def decode_to_working(rgb: np.ndarray, profile_id: str) -> np.ndarray:
    """Decode camera code RGB to scene-linear Rec.709 primaries, float32."""
    try:
        profile = PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown input camera profile {profile_id!r}") from exc
    if profile_id == "legacy":
        return np.asarray(rgb, dtype=np.float32).copy()
    colour = _colour()
    to_xyz = _rgb_to_xyz(profile.gamut)
    xyz_to_working = np.asarray(colour.RGB_COLOURSPACES["ITU-R BT.709"].matrix_XYZ_to_RGB, dtype=np.float32)
    matrix = xyz_to_working @ to_xyz
    return _apply_tiled(rgb, lambda tile: _matrix_transform(_curve(tile, profile.curve, False), matrix))


def encode_from_working(linear709: np.ndarray, profile_id: str) -> np.ndarray:
    """Encode linear Rec.709 RGB into the selected camera code RGB profile."""
    try:
        profile = PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown output camera profile {profile_id!r}") from exc
    if profile_id == "legacy":
        return np.asarray(linear709, dtype=np.float32).copy()
    colour = _colour()
    from_xyz = np.asarray(colour.RGB_COLOURSPACES["ITU-R BT.709"].matrix_RGB_to_XYZ, dtype=np.float32)
    matrix = np.linalg.inv(_rgb_to_xyz(profile.gamut)) @ from_xyz
    return _apply_tiled(linear709, lambda tile: _curve(_matrix_transform(tile, matrix), profile.curve, True))
