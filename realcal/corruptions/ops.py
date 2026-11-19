"""The 15 fundus corruption operators (plus 'pixelate' for baseline B2).

Every operator works on float32 RGB in [0, 1] together with a boolean field-of-view mask and returns
``(image, mask)``. Pixels outside the (returned) mask are set to black by ``apply_op``. Blurs use
normalised convolution inside the FOV so that the black border does not bleed into the retina.
Spatial parameters are given at 224 px and multiplied by ``scale = size / 224``.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

Op = Callable[..., tuple[np.ndarray, np.ndarray]]


def _norm_conv(x: np.ndarray, m: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    mf = m.astype(np.float32)
    num = cv2.filter2D(x * mf[..., None], -1, kernel, borderType=cv2.BORDER_CONSTANT)
    den = cv2.filter2D(mf, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    out = num / np.maximum(den, 1e-6)[..., None]
    return np.where(m[..., None], out, x)


def _gauss_norm(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    mf = m.astype(np.float32)
    num = cv2.GaussianBlur(x * mf[..., None], (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    den = cv2.GaussianBlur(mf, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    return num / np.maximum(den, 1e-6)[..., None]


def _fov_geometry(m: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        h, w = m.shape
        return w / 2, h / 2, min(h, w) / 2
    cx, cy = xs.mean(), ys.mean()
    r = math.sqrt(m.sum() / math.pi)
    return float(cx), float(cy), float(r)


def defocus(x, m, radius, rng, scale):  # noqa: ARG001
    r = max(0.5, radius * scale)
    k = int(math.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    ker = ((xx ** 2 + yy ** 2) <= r ** 2 + 1e-6).astype(np.float32)
    ker = cv2.GaussianBlur(ker, (3, 3), 0.5) if k >= 1 else ker
    ker /= ker.sum()
    return _norm_conv(x, m, ker), m


def motion(x, m, length, rng, scale):
    L = max(2, int(round(length * scale)))
    ang = float(rng.uniform(0, 180))
    ker = np.zeros((L, L), np.float32)
    ker[L // 2, :] = 1.0
    R = cv2.getRotationMatrix2D(((L - 1) / 2, (L - 1) / 2), ang, 1.0)
    ker = cv2.warpAffine(ker, R, (L, L))
    ker /= max(ker.sum(), 1e-6)
    return _norm_conv(x, m, ker), m


def gaussian_noise(x, m, sigma, rng, scale):  # noqa: ARG001
    return x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32), m


def shot_noise(x, m, lam, rng, scale):  # noqa: ARG001
    return (rng.poisson(np.clip(x, 0, 1) * lam) / float(lam)).astype(np.float32), m


def _hsv_shift(x, delta):
    hsv = cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] + delta, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def over_exposure(x, m, a, rng, scale):  # noqa: ARG001
    return _hsv_shift(x, a), m


def under_exposure(x, m, a, rng, scale):  # noqa: ARG001
    return _hsv_shift(x, -a), m


def uneven_illumination(x, m, s, rng, scale):  # noqa: ARG001
    cx, cy, r = _fov_geometry(m)
    ang = rng.uniform(0, 2 * math.pi)
    off = rng.uniform(0, 0.25) * r
    ccx, ccy = cx + off * math.cos(ang), cy + off * math.sin(ang)
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2)
    dmax = max(float(d[m].max()) if m.any() else r, 1.0)
    fall = 1.0 - s * np.clip(d / dmax, 0, 1) ** 2
    return x * fall[..., None].astype(np.float32), m


def haze(x, m, t, rng, scale, sigma_frac=0.08, warm=(0.80, 0.72, 0.62), brighten=0.35):  # noqa: ARG001
    sigma = max(1.0, sigma_frac * x.shape[0])
    blur = _gauss_norm(x, m, sigma)
    warm_a = np.asarray(warm, np.float32).reshape(1, 1, 3)
    veil = (1 - brighten) * blur + brighten * warm_a
    return (1 - t) * x + t * veil, m


def contrast_loss(x, m, c, rng, scale):  # noqa: ARG001
    mu = float(x[m].mean()) if m.any() else float(x.mean())
    return mu + c * (x - mu), m


def gamma_shift(x, m, g, rng, scale):  # noqa: ARG001
    sign = 1.0 if rng.random() < 0.5 else -1.0
    gamma = 2.0 ** (sign * g)
    return np.power(np.clip(x, 0, 1), gamma), m


def colour_cast(x, m, a, rng, scale):  # noqa: ARG001
    gains = 1.0 + rng.uniform(-a, a, size=3).astype(np.float32)
    return x * gains.reshape(1, 1, 3), m


def jpeg(x, m, q, rng, scale):  # noqa: ARG001
    u8 = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return dec.astype(np.float32) / 255.0, m


def low_resolution(x, m, f, rng, scale):  # noqa: ARG001
    h, w = x.shape[:2]
    sh, sw = max(2, int(round(h / f))), max(2, int(round(w / f)))
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), m


def pixelate(x, m, f, rng, scale):  # noqa: ARG001
    h, w = x.shape[:2]
    sh, sw = max(2, int(round(h / f))), max(2, int(round(w / f)))
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), m


def off_centre(x, m, frac, rng, scale):  # noqa: ARG001
    _, _, r = _fov_geometry(m)
    shift = frac * 2 * r
    ang = rng.uniform(0, 2 * math.pi)
    dx, dy = shift * math.cos(ang), shift * math.sin(ang)
    h, w = m.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    xs = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ms = cv2.warpAffine(m.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return xs, ms


def occlusion(x, m, area, rng, scale):  # noqa: ARG001
    cx, cy, r = _fov_geometry(m)
    h, w = m.shape
    fov_area = float(m.sum()) if m.any() else float(h * w)
    target = area * fov_area
    alpha = np.zeros((h, w), np.float32)
    if rng.random() < 0.5:
        # bright soft-edged blob centred on the FOV border
        rad = math.sqrt(target / math.pi) * 1.4  # half of the disc lies outside the FOV
        ang = rng.uniform(0, 2 * math.pi)
        bx, by = cx + r * math.cos(ang), cy + r * math.sin(ang)
        cv2.circle(alpha, (int(bx), int(by)), int(max(2, rad)), 1.0, thickness=-1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, rad * 0.25))
        colour = np.asarray([0.97, 0.93, 0.86], np.float32) * rng.uniform(0.9, 1.0)
    else:
        # 3-8 dark curved eyelash-like streaks from the upper or lower edge
        n = int(rng.integers(3, 9))
        top = rng.random() < 0.5
        length = r * rng.uniform(0.5, 0.9)
        thick = max(1, int(round(target / max(n * length, 1.0))))
        for _ in range(n):
            x0 = cx + rng.uniform(-0.8, 0.8) * r
            y0 = cy - r if top else cy + r
            dirn = 1 if top else -1
            x2 = x0 + rng.uniform(-0.4, 0.4) * length
            y2 = y0 + dirn * length
            x1 = (x0 + x2) / 2 + rng.uniform(-0.3, 0.3) * length
            y1 = (y0 + y2) / 2
            tt = np.linspace(0, 1, 40)
            px = (1 - tt) ** 2 * x0 + 2 * (1 - tt) * tt * x1 + tt ** 2 * x2
            py = (1 - tt) ** 2 * y0 + 2 * (1 - tt) * tt * y1 + tt ** 2 * y2
            pts = np.stack([px, py], 1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(alpha, [pts], False, 1.0, thickness=thick, lineType=cv2.LINE_AA)
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.7, thick * 0.3))
        colour = np.asarray([0.04, 0.03, 0.03], np.float32)
    alpha = np.clip(alpha, 0, 1)[..., None] * m[..., None]
    return x * (1 - alpha) + colour.reshape(1, 1, 3) * alpha, m


OPS: dict[str, Op] = {
    "defocus": defocus, "motion": motion, "gaussian_noise": gaussian_noise, "shot_noise": shot_noise,
    "over_exposure": over_exposure, "under_exposure": under_exposure,
    "uneven_illumination": uneven_illumination, "haze": haze, "contrast_loss": contrast_loss,
    "gamma_shift": gamma_shift, "colour_cast": colour_cast, "jpeg": jpeg, "low_resolution": low_resolution,
    "off_centre": off_centre, "occlusion": occlusion, "pixelate": pixelate,
}


def run_op(op: str, img_u8: np.ndarray, mask: np.ndarray, param: float, rng: np.random.Generator,
           spatial: bool, suite_cfg=None) -> np.ndarray:
    """Apply operator ``op`` with an explicit parameter; returns uint8 RGB (outside FOV black)."""
    size = img_u8.shape[0]
    scale = size / 224.0 if spatial else 1.0
    x = img_u8.astype(np.float32) / 255.0
    m = mask.astype(bool)
    if op == "haze" and suite_cfg is not None:
        y, mo = haze(x, m, param, rng, scale, float(suite_cfg.haze_blur_sigma_frac),
                     tuple(suite_cfg.haze_warm_grey), float(suite_cfg.haze_brighten))
    else:
        y, mo = OPS[op](x, m, param, rng, scale)
    y = np.clip(y, 0, 1) * mo[..., None]
    return (y * 255 + 0.5).astype(np.uint8)


def apply_op(suite_cfg, name: str, severity: int, img_u8: np.ndarray, mask: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    """Apply main-suite corruption ``name`` at severity 1-5 (used for corruption-augmented training)."""
    c = suite_cfg.corruptions[name]
    return run_op(name, img_u8, mask, float(c.params[severity - 1]), rng, bool(c.spatial), suite_cfg)
