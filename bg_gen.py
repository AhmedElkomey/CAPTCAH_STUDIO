import os
import math
import secrets
import random
from typing import Optional, Tuple, List, Dict, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# Core helpers
# ============================================================

CUSTOM_PHOTO_DIR = None
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_CUSTOM_PHOTO_CACHE = {"dir": None, "stamp": None, "files": []}


def _custom_photo_files() -> List[str]:
    if not CUSTOM_PHOTO_DIR or not os.path.isdir(CUSTOM_PHOTO_DIR):
        return []
    try:
        stamp = os.path.getmtime(CUSTOM_PHOTO_DIR)
    except OSError:
        return []
    if (
        _CUSTOM_PHOTO_CACHE["dir"] == CUSTOM_PHOTO_DIR
        and _CUSTOM_PHOTO_CACHE["stamp"] == stamp
    ):
        return list(_CUSTOM_PHOTO_CACHE["files"])
    try:
        files = [
            name
            for name in os.listdir(CUSTOM_PHOTO_DIR)
            if os.path.splitext(name.lower())[1] in PHOTO_EXTENSIONS
        ]
    except OSError:
        return []
    _CUSTOM_PHOTO_CACHE["dir"] = CUSTOM_PHOTO_DIR
    _CUSTOM_PHOTO_CACHE["stamp"] = stamp
    _CUSTOM_PHOTO_CACHE["files"] = list(files)
    return files


def _has_custom_photo_dir() -> bool:
    return len(_custom_photo_files()) > 0

def _to_float01(img_rgb: Image.Image) -> np.ndarray:
    return np.asarray(img_rgb, dtype=np.float32) / 255.0


def _from_float01(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def overlay_blend(base_rgb: Image.Image, top_rgb: Image.Image, opacity: float = 0.35) -> Image.Image:
    """Photoshop-like Overlay blend with opacity."""
    b = _to_float01(base_rgb)
    t = _to_float01(top_rgb)

    out = np.where(b <= 0.5, 2.0 * b * t, 1.0 - 2.0 * (1.0 - b) * (1.0 - t))
    mixed = b + opacity * (out - b)
    return _from_float01(mixed)


def soft_light_blend(base_rgb: Image.Image, top_rgb: Image.Image, opacity: float = 0.35) -> Image.Image:
    """Soft light blend (approx), mixed with opacity."""
    b = _to_float01(base_rgb)
    t = _to_float01(top_rgb)

    out = (1.0 - 2.0 * t) * (b * b) + 2.0 * t * b
    mixed = b + opacity * (out - b)
    return _from_float01(mixed)


def fbm_noise(width: int, height: int, rng: np.random.Generator,
              octaves: int = 6, persistence: float = 0.55) -> np.ndarray:
    """
    Fractal-ish smooth noise by summing upscaled random grids (value noise).
    Returns float32 array in [0, 1].
    """
    acc = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    freq = 1

    for _ in range(octaves):
        h2 = max(1, height // freq)
        w2 = max(1, width // freq)
        
        # Avoid float overhead during generation
        small = rng.integers(0, 256, size=(h2, w2), dtype=np.uint8)

        layer = Image.fromarray(small, mode="L").resize(
            (width, height), resample=Image.BICUBIC
        )

        acc += amp * np.asarray(layer, dtype=np.float32)
        amp *= persistence
        freq *= 2

    # Fast normalization
    acc_min, acc_max = acc.min(), acc.max()
    if acc_max > acc_min:
        acc = (acc - acc_min) / (acc_max - acc_min)
    else:
        acc.fill(0.0)
    return acc


def ridged_noise(n: np.ndarray) -> np.ndarray:
    """Turns noise into ridged noise (useful for stone/leather)."""
    r = 1.0 - np.abs(2.0 * n - 1.0)
    r = (r - r.min()) / (r.max() - r.min() + 1e-6)
    return r


def radial_vignette_mask(width: int, height: int, strength: float = 0.55, power: float = 1.7) -> Image.Image:
    """L mask: edges brighter (more subtraction = darker edges)."""
    y, x = np.ogrid[:height, :width]
    cx, cy = width / 2.0, height / 2.0
    
    # Avoid sqrt inside the power step by halving power for squared distances
    r_sq = (x - cx) ** 2 + (y - cy) ** 2
    r_sq /= (r_sq.max() + 1e-6)
    
    m = np.clip((r_sq ** (power / 2.0)) * 255.0 * strength, 0, 255).astype(np.uint8)
    blur = max(1, int(min(width, height) * 0.03))
    return Image.fromarray(m, mode="L").filter(ImageFilter.GaussianBlur(radius=blur))


def palette_map(v: np.ndarray, stops: List[Tuple[float, Tuple[int, int, int]]]) -> Image.Image:
    """
    Map float array v in [0,1] to RGB using color stops.
    stops: [(pos0..1, (r,g,b)), ...] sorted by pos.
    """
    v = np.clip(v, 0.0, 1.0)
    stops = sorted(stops, key=lambda s: s[0])

    xp = np.array([s[0] for s in stops], dtype=np.float32)
    fp_r = np.array([s[1][0] for s in stops], dtype=np.float32)
    fp_g = np.array([s[1][1] for s in stops], dtype=np.float32)
    fp_b = np.array([s[1][2] for s in stops], dtype=np.float32)

    # Use extremely fast np.interp on 1D representations
    v_flat = v.ravel()
    r = np.interp(v_flat, xp, fp_r).reshape(v.shape)
    g = np.interp(v_flat, xp, fp_g).reshape(v.shape)
    b = np.interp(v_flat, xp, fp_b).reshape(v.shape)

    out = np.stack((r, g, b), axis=-1)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def add_grain_seeded(width: int, height: int, rng: np.random.Generator, strength: float = 0.12) -> Image.Image:
    g = rng.standard_normal(size=(height, width), dtype=np.float32) * 0.18 + 0.5
    g = np.clip(g, 0.0, 1.0)
    g = (g * 255.0 * strength).astype(np.uint8)
    return Image.fromarray(g, "L").filter(ImageFilter.GaussianBlur(0.35))


def add_fibers(width: int, height: int, rng: np.random.Generator, intensity: float = 0.18) -> Image.Image:
    base = (rng.random((max(1, height // 8), max(1, width // 120)), dtype=np.float32) * 255).astype(np.uint8)
    img = Image.fromarray(base, "L").resize((width, height), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    img = img.rotate(float(rng.uniform(-14, 14)), resample=Image.BICUBIC, expand=False)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip((arr - 0.5) * 1.8 + 0.5, 0, 1)
    arr = (arr * 255.0 * intensity).astype(np.uint8)
    return Image.fromarray(arr, "L")


def add_speckles(width: int, height: int, rng: np.random.Generator, amount: float = 0.003) -> Image.Image:
    arr = np.zeros((height, width), dtype=np.uint8)
    n = int(width * height * amount)
    if n <= 0:
        return Image.fromarray(arr, "L")
    ys = rng.integers(0, height, size=n)
    xs = rng.integers(0, width, size=n)
    vals = rng.integers(110, 255, size=n, dtype=np.uint8)
    arr[ys, xs] = vals
    return Image.fromarray(arr, "L").filter(ImageFilter.GaussianBlur(radius=0.6))


# ============================================================
# Distortion (mesh warp)  -> real deformation
# ============================================================

def mesh_warp(img: Image.Image, rng: np.random.Generator, grid_size: int = 6, magnitude: float = 6.0) -> Image.Image:
    if cv2 is None:
        # Fallback if OpenCV is not available.
        return _mesh_warp_pil(img, rng, grid_size, magnitude)

    w, h = img.size
    tiles_x = max(2, int(round((w / min(w, h)) * grid_size)))
    tiles_y = max(2, grid_size)
    nx, ny = tiles_x + 1, tiles_y + 1
    
    # Generate displacement grid
    dx = rng.uniform(-magnitude, magnitude, size=(ny, nx)).astype(np.float32)
    dy = rng.uniform(-magnitude, magnitude, size=(ny, nx)).astype(np.float32)
    
    # Pin edges
    dx[0, :] = 0; dx[-1, :] = 0; dx[:, 0] = 0; dx[:, -1] = 0
    dy[0, :] = 0; dy[-1, :] = 0; dy[:, 0] = 0; dy[:, -1] = 0
    
    # Resize to full image size smoothly
    flow_x = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    flow_y = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)
    
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + flow_x
    map_y = yy + flow_y
    
    # Remap
    arr = np.array(img)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    return Image.fromarray(warped, mode=img.mode)

def _mesh_warp_pil(img: Image.Image, rng: np.random.Generator, grid_size: int = 6, magnitude: float = 6.0) -> Image.Image:
    w, h = img.size
    tiles_x = max(2, int(round((w / min(w, h)) * grid_size)))
    tiles_y = max(2, grid_size)

    dx = w / tiles_x
    dy = h / tiles_y

    points = []
    for j in range(tiles_y + 1):
        row = []
        for i in range(tiles_x + 1):
            x = i * dx
            y = j * dy
            border = (i == 0 or j == 0 or i == tiles_x or j == tiles_y)
            mag = magnitude * (0.25 if border else 1.0)
            row.append((x + float(rng.uniform(-mag, mag)), y + float(rng.uniform(-mag, mag))))
        points.append(row)

    mesh = []
    for j in range(tiles_y):
        for i in range(tiles_x):
            x0 = int(round(i * dx))
            y0 = int(round(j * dy))
            x1 = int(round((i + 1) * dx))
            y1 = int(round((j + 1) * dy))
            p00 = points[j][i]
            p10 = points[j][i + 1]
            p11 = points[j + 1][i + 1]
            p01 = points[j + 1][i]
            quad = (p00[0], p00[1], p10[0], p10[1], p11[0], p11[1], p01[0], p01[1])
            mesh.append(((x0, y0, x1, y1), quad))

    return img.transform((w, h), Image.MESH, mesh, resample=Image.BICUBIC)


# ============================================================
# Optional “synthetic spice” overlay (keep subtle)
# ============================================================

def generate_scribble_layer(width: int, height: int, rnd: random.Random) -> Image.Image:
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    num = rnd.randint(40, 70)
    for _ in range(num):
        color = (rnd.randint(40, 210), rnd.randint(40, 210), rnd.randint(40, 210), 255)
        t = rnd.choice(["line", "curve", "zigzag", "circle"])

        if t == "line":
            x1, y1 = rnd.randint(-50, width + 50), rnd.randint(-50, height + 50)
            x2, y2 = rnd.randint(-50, width + 50), rnd.randint(-50, height + 50)
            draw.line([x1, y1, x2, y2], fill=color, width=rnd.randint(1, 4))
        elif t == "curve":
            x1, y1 = rnd.randint(-50, width), rnd.randint(-50, height)
            x2, y2 = rnd.randint(x1 + 10, width + 50), rnd.randint(y1 + 10, height + 50)
            start = rnd.randint(0, 180)
            end = start + rnd.randint(60, 200)
            draw.arc([x1, y1, x2, y2], start=start, end=end, fill=color, width=rnd.randint(1, 4))
        elif t == "zigzag":
            pts = [(rnd.randint(0, width), rnd.randint(0, height))]
            for _ in range(rnd.randint(2, 5)):
                x = pts[-1][0] + rnd.randint(-35, 35)
                y = pts[-1][1] + rnd.randint(-25, 25)
                pts.append((x, y))
            draw.line(pts, fill=color, width=rnd.randint(1, 3))
        else:
            r = rnd.randint(3, 12)
            cx, cy = rnd.randint(0, width), rnd.randint(0, height)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=rnd.randint(1, 4))

    return layer.filter(ImageFilter.GaussianBlur(radius=rnd.uniform(0.6, 1.8)))


def composite_with_scribbles(bg: Image.Image, scrib: Image.Image, opacity: float = 0.10) -> Image.Image:
    s = scrib.copy()
    a = s.split()[-1].point(lambda p: int(p * opacity))
    s.putalpha(a)
    return Image.alpha_composite(bg, s)


# ============================================================
# Texture generators (WIDE variety)
# Each returns RGBA
# ============================================================

def tex_paper(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n = fbm_noise(width, height, rng, octaves=6, persistence=0.55)
    n = np.clip((n - 0.5) * 1.15 + 0.5, 0, 1)

    base = palette_map(n, [
        (0.0, (235, 232, 224)),
        (0.5, (245, 243, 238)),
        (1.0, (228, 224, 212)),
    ])

    fibers = add_fibers(width, height, rng, intensity=float(rng.uniform(0.18, 0.30)))
    f_rgb = Image.merge("RGB", (fibers, fibers, fibers))
    out = soft_light_blend(base, f_rgb, opacity=0.35)

    speck = add_speckles(width, height, rng, amount=float(rng.uniform(0.0015, 0.004)))
    s_rgb = Image.merge("RGB", (speck, speck, speck))
    out = overlay_blend(out, s_rgb, opacity=0.18)

    vign = radial_vignette_mask(width, height, strength=0.38, power=1.6)
    out = ImageChops.subtract(out, Image.merge("RGB", (vign, vign, vign)))

    grain = add_grain_seeded(width, height, rng, strength=0.10)
    out = overlay_blend(out, Image.merge("RGB", (grain, grain, grain)), opacity=0.12)

    return out.convert("RGBA")


def tex_parchment(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n = fbm_noise(width, height, rng, octaves=6, persistence=0.55)
    blotch = fbm_noise(width, height, rng, octaves=4, persistence=0.65)
    blotch = np.clip((blotch - 0.45) * 1.8 + 0.5, 0, 1)

    base = palette_map(n, [
        (0.0, (210, 190, 145)),
        (0.5, (235, 218, 170)),
        (1.0, (195, 170, 120)),
    ])

    stains = palette_map(blotch, [
        (0.0, (120, 85, 40)),
        (0.5, (180, 130, 70)),
        (1.0, (250, 235, 200)),
    ])
    out = soft_light_blend(base, stains, opacity=0.35)

    # darker edges like old parchment
    vign = radial_vignette_mask(width, height, strength=0.70, power=1.35)
    out = ImageChops.subtract(out, Image.merge("RGB", (vign, vign, vign)))

    # fibers and dust
    fibers = add_fibers(width, height, rng, intensity=float(rng.uniform(0.16, 0.28)))
    out = overlay_blend(out, Image.merge("RGB", (fibers, fibers, fibers)), opacity=0.22)

    speck = add_speckles(width, height, rng, amount=float(rng.uniform(0.002, 0.006)))
    out = overlay_blend(out, Image.merge("RGB", (speck, speck, speck)), opacity=0.28)

    return out.convert("RGBA")


def tex_canvas(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n = fbm_noise(width, height, rng, octaves=5, persistence=0.55)
    n = np.clip((n - 0.5) * 1.10 + 0.5, 0, 1)

    base = palette_map(n, [
        (0.0, (230, 225, 215)),
        (0.5, (245, 243, 238)),
        (1.0, (222, 216, 205)),
    ])

    # weave pattern (sinusoidal grid)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    period = float(rng.uniform(8.0, 14.0))  # small for 350x100
    weave = (np.sin(2 * math.pi * xx / period) * np.sin(2 * math.pi * yy / period)) * 0.5 + 0.5
    weave = np.clip((weave - 0.5) * 1.6 + 0.5, 0, 1)
    weave_img = palette_map(weave, [(0.0, (80, 80, 80)), (1.0, (220, 220, 220))]).convert("L")
    weave_img = weave_img.filter(ImageFilter.GaussianBlur(0.4)).filter(ImageFilter.EMBOSS)
    out = overlay_blend(base, Image.merge("RGB", (weave_img, weave_img, weave_img)), opacity=0.22)

    fibers = add_fibers(width, height, rng, intensity=float(rng.uniform(0.20, 0.32)))
    out = soft_light_blend(out, Image.merge("RGB", (fibers, fibers, fibers)), opacity=0.28)

    grain = add_grain_seeded(width, height, rng, strength=0.12)
    out = overlay_blend(out, Image.merge("RGB", (grain, grain, grain)), opacity=0.14)

    return out.convert("RGBA")


def tex_concrete(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n = fbm_noise(width, height, rng, octaves=6, persistence=0.55)
    r = ridged_noise(n)
    r = np.clip((r - 0.5) * 1.6 + 0.5, 0, 1)

    base = palette_map(r, [
        (0.0, (70, 70, 70)),
        (0.5, (135, 135, 135)),
        (1.0, (210, 210, 210)),
    ])

    speck = add_speckles(width, height, rng, amount=float(rng.uniform(0.004, 0.010)))
    out = overlay_blend(base, Image.merge("RGB", (speck, speck, speck)), opacity=0.35)

    vign = radial_vignette_mask(width, height, strength=0.55, power=1.55)
    out = ImageChops.subtract(out, Image.merge("RGB", (vign, vign, vign)))

    return out.convert("RGBA")


def tex_marble(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    # veins: sin(x + turbulence)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t1 = fbm_noise(width, height, rng, octaves=5, persistence=0.58)
    t2 = fbm_noise(width, height, rng, octaves=3, persistence=0.62)

    scale = float(rng.uniform(10.0, 22.0))  # tuned for small images
    veins = np.sin((xx / scale) + (t1 * 3.0) + (t2 * 1.2))
    veins = (veins * 0.5 + 0.5)
    veins = np.clip((veins - 0.5) * 1.6 + 0.5, 0, 1)

    base = palette_map(veins, [
        (0.0, (25, 25, 28)),
        (0.35, (140, 140, 150)),
        (0.65, (225, 225, 230)),
        (1.0, (245, 245, 248)),
    ])

    # add cloudy body
    body = fbm_noise(width, height, rng, octaves=5, persistence=0.55)
    body_img = palette_map(body, [(0.0, (120, 120, 130)), (1.0, (250, 250, 252))])
    out = soft_light_blend(base, body_img, opacity=0.22)

    # slight polish highlight
    vign = radial_vignette_mask(width, height, strength=0.30, power=1.9)
    out = ImageChops.subtract(out, Image.merge("RGB", (vign, vign, vign)))

    return out.convert("RGBA")


def tex_wood(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = fbm_noise(width, height, rng, octaves=5, persistence=0.58)

    # wood rings along x with turbulence
    scale = float(rng.uniform(8.0, 16.0))
    rings = np.sin((xx / scale) + (t * 6.0))
    rings = rings * 0.5 + 0.5
    rings = np.clip((rings - 0.5) * 1.7 + 0.5, 0, 1)

    base = palette_map(rings, [
        (0.0, (60, 32, 14)),
        (0.35, (120, 70, 35)),
        (0.65, (170, 115, 60)),
        (1.0, (235, 190, 130)),
    ])

    # pores / small noise
    pores = fbm_noise(width, height, rng, octaves=3, persistence=0.62)
    pores = np.clip((pores - 0.5) * 2.0 + 0.5, 0, 1)
    pores_img = palette_map(pores, [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))])
    out = overlay_blend(base, pores_img, opacity=0.18)

    return out.convert("RGBA")


def tex_brushed_metal(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    # Make horizontal streaks by generating 1D noise then expanding
    line = rng.random(width, dtype=np.float32)
    # smooth the line
    k = int(max(3, width * 0.03))
    kernel = np.ones(k, dtype=np.float32) / k
    line = np.convolve(line, kernel, mode="same")
    line = (line - line.min()) / (line.max() - line.min() + 1e-6)

    streak = np.tile(line[None, :], (height, 1))
    # add gentle vertical variation
    v = fbm_noise(width, height, rng, octaves=3, persistence=0.6)
    streak = np.clip(streak * 0.85 + v * 0.15, 0, 1)

    base = palette_map(streak, [
        (0.0, (40, 40, 45)),
        (0.5, (160, 165, 175)),
        (1.0, (235, 240, 248)),
    ])

    # subtle emboss
    l = base.convert("L").filter(ImageFilter.GaussianBlur(0.7)).filter(ImageFilter.EMBOSS)
    out = overlay_blend(base, Image.merge("RGB", (l, l, l)), opacity=0.20)

    return out.convert("RGBA")


def tex_leather(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n = fbm_noise(width, height, rng, octaves=6, persistence=0.55)
    r = ridged_noise(n)
    r = np.clip((r - 0.5) * 1.8 + 0.5, 0, 1)

    # pore shaping using min/max filters
    l = Image.fromarray((r * 255).astype(np.uint8), "L")
    l = l.filter(ImageFilter.GaussianBlur(0.8))
    l = l.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    l = l.filter(ImageFilter.EMBOSS)

    # brown-ish palette
    base = palette_map(r, [
        (0.0, (35, 18, 10)),
        (0.45, (95, 55, 30)),
        (0.75, (140, 95, 55)),
        (1.0, (200, 160, 110)),
    ])

    out = overlay_blend(base, Image.merge("RGB", (l, l, l)), opacity=0.25)
    return out.convert("RGBA")


def tex_denim(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    period = float(rng.uniform(6.0, 12.0))
    twill = np.sin(2 * math.pi * (xx + yy * 0.9) / period) * 0.5 + 0.5
    twill = np.clip((twill - 0.5) * 1.6 + 0.5, 0, 1)

    body = fbm_noise(width, height, rng, octaves=5, persistence=0.58)
    v = np.clip(twill * 0.65 + body * 0.35, 0, 1)

    base = palette_map(v, [
        (0.0, (10, 18, 45)),
        (0.5, (35, 70, 140)),
        (1.0, (160, 200, 240)),
    ])

    # add subtle thread grain
    grain = add_grain_seeded(width, height, rng, strength=0.13)
    out = overlay_blend(base, Image.merge("RGB", (grain, grain, grain)), opacity=0.16)
    return out.convert("RGBA")


def tex_watercolor_wash(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n = fbm_noise(width, height, rng, octaves=6, persistence=0.60)
    blobs = fbm_noise(width, height, rng, octaves=3, persistence=0.70)

    # soft edges like pigment pooling
    pool = np.clip((blobs - 0.45) * 1.9 + 0.5, 0, 1)
    pool_img = Image.fromarray((pool * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(2.0))
    pool = np.asarray(pool_img, dtype=np.float32) / 255.0

    # pick a random pastel-ish palette
    base_color = np.array([rng.integers(160, 240), rng.integers(160, 240), rng.integers(160, 240)], dtype=np.float32)
    accent = np.array([rng.integers(40, 160), rng.integers(40, 160), rng.integers(40, 160)], dtype=np.float32)

    v = np.clip(n * 0.7 + pool * 0.3, 0, 1)

    rgb = (base_color[None, None, :] * (1.0 - v[:, :, None]) + accent[None, None, :] * (v[:, :, None])) / 255.0
    img = _from_float01(np.clip(rgb, 0, 1))

    # paper under it
    paper = tex_paper(width, height, rng).convert("RGB")
    out = soft_light_blend(paper, img, opacity=0.55)

    # pigment edges
    edge = np.clip((pool - 0.5) * 2.4 + 0.5, 0, 1)
    edge_img = Image.fromarray((edge * 255).astype(np.uint8), "L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(1.2))
    out = overlay_blend(out, Image.merge("RGB", (edge_img, edge_img, edge_img)), opacity=0.18)

    return out.convert("RGBA")


def tex_clouds_sky(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n = fbm_noise(width, height, rng, octaves=6, persistence=0.55)
    n = np.clip((n - 0.45) * 1.7 + 0.5, 0, 1)

    sky = palette_map(n, [
        (0.0, (20, 60, 120)),
        (0.45, (90, 160, 230)),
        (0.75, (200, 225, 250)),
        (1.0, (255, 255, 255)),
    ])

    # gentle top-to-bottom gradient
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    grad = palette_map(y.repeat(width, axis=1), [
        (0.0, (15, 40, 95)),
        (1.0, (185, 220, 255)),
    ])
    out = soft_light_blend(sky, grad, opacity=0.25)

    return out.convert("RGBA")


def tex_bokeh(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    # smooth colored gradient background
    c1 = (int(rng.integers(10, 90)), int(rng.integers(10, 90)), int(rng.integers(10, 90)))
    c2 = (int(rng.integers(140, 240)), int(rng.integers(140, 240)), int(rng.integers(140, 240)))
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    base_arr = (np.array(c1, dtype=np.float32) * (1 - y) + np.array(c2, dtype=np.float32) * y)
    base = Image.fromarray(np.repeat(base_arr[:, None, :], width, axis=1).astype(np.uint8), "RGB")

    # circles
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    count = int(rng.integers(12, 28))
    for _ in range(count):
        r = int(rng.integers(max(6, min(width, height)//12), max(14, min(width, height)//4)))
        cx = int(rng.integers(-r, width + r))
        cy = int(rng.integers(-r, height + r))
        col = (int(rng.integers(160, 255)), int(rng.integers(160, 255)), int(rng.integers(160, 255)), int(rng.integers(35, 90)))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    layer = layer.filter(ImageFilter.GaussianBlur(radius=4.0))
    out = Image.alpha_composite(base.convert("RGBA"), layer)

    vign = radial_vignette_mask(width, height, strength=0.45, power=1.6)
    out_rgb = ImageChops.subtract(out.convert("RGB"), Image.merge("RGB", (vign, vign, vign)))
    return out_rgb.convert("RGBA")


def tex_waves(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    # Ocean/water waves with sine and noise
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = fbm_noise(width, height, rng, octaves=4, persistence=0.55)
    
    scale = float(rng.uniform(15.0, 35.0))
    # Distort coordinates with noise
    waves = np.sin((xx / scale) + (t * 5.0)) * np.cos((yy / scale) + (t * 5.0))
    waves = waves * 0.5 + 0.5
    
    # Add turbulent layer
    turb = fbm_noise(width, height, rng, octaves=5, persistence=0.6)
    v = np.clip(waves * 0.7 + turb * 0.3, 0, 1)

    base = palette_map(v, [
        (0.0, (10, 30, 80)),
        (0.4, (20, 80, 150)),
        (0.8, (80, 170, 220)),
        (1.0, (230, 245, 255)),
    ])
    
    vign = radial_vignette_mask(width, height, strength=0.3, power=1.5)
    out = ImageChops.subtract(base, Image.merge("RGB", (vign, vign, vign)))
    return out.convert("RGBA")


def tex_tree_bark(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    # Stretch noise vertically
    n_small = fbm_noise(width, max(1, height // 4), rng, octaves=4, persistence=0.6)
    n_img = Image.fromarray((n_small * 255).astype(np.uint8), "L").resize((width, height), Image.BICUBIC)
    n = np.asarray(n_img, dtype=np.float32) / 255.0
    
    r = ridged_noise(n)
    r = np.clip((r - 0.5) * 1.5 + 0.5, 0, 1)

    base = palette_map(r, [
        (0.0, (20, 15, 10)),
        (0.3, (40, 30, 20)),
        (0.7, (80, 65, 50)),
        (1.0, (120, 110, 95)),
    ])
    
    grain = add_grain_seeded(width, height, rng, strength=0.15)
    out = overlay_blend(base, Image.merge("RGB", (grain, grain, grain)), opacity=0.20)
    return out.convert("RGBA")


def tex_sand(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    # Low frequency dunes
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    scale = float(rng.uniform(20.0, 40.0))
    dunes = np.sin((xx / scale) + np.sin(yy / scale * 0.5)) * 0.5 + 0.5
    
    n = fbm_noise(width, height, rng, octaves=3, persistence=0.55)
    v = np.clip(dunes * 0.6 + n * 0.4, 0, 1)
    
    base = palette_map(v, [
        (0.0, (180, 150, 110)),
        (0.5, (210, 185, 140)),
        (1.0, (240, 220, 180)),
    ])
    
    grain = add_grain_seeded(width, height, rng, strength=0.25)
    out = soft_light_blend(base, Image.merge("RGB", (grain, grain, grain)), opacity=0.35)
    return out.convert("RGBA")


def tex_camo(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n1 = fbm_noise(width, height, rng, octaves=4, persistence=0.55)
    n2 = fbm_noise(width, height, rng, octaves=4, persistence=0.55)
    
    # Sharp thresholds
    m1 = (n1 > 0.6).astype(np.float32)
    m2 = (n2 > 0.55).astype(np.float32)
    m3 = (n1 < 0.35).astype(np.float32)
    
    base = np.full((height, width, 3), (95, 115, 70), dtype=np.float32) # Base green
    
    brown = np.array((85, 65, 45), dtype=np.float32)
    dark = np.array((45, 50, 40), dtype=np.float32)
    light = np.array((160, 165, 130), dtype=np.float32)
    
    base = base * (1 - m1[:, :, None]) + brown * m1[:, :, None]
    base = base * (1 - m2[:, :, None]) + dark * m2[:, :, None]
    base = base * (1 - m3[:, :, None]) + light * m3[:, :, None]
    
    img = Image.fromarray(np.clip(base + 0.5, 0, 255).astype(np.uint8), "RGB")
    grain = add_grain_seeded(width, height, rng, strength=0.1)
    out = overlay_blend(img, Image.merge("RGB", (grain, grain, grain)), opacity=0.1)
    return out.convert("RGBA")


def tex_rust(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    metal = fbm_noise(width, height, rng, octaves=3, persistence=0.5)
    base = palette_map(metal, [(0.0, (60, 65, 70)), (1.0, (120, 125, 130))])
    
    rust_n = fbm_noise(width, height, rng, octaves=6, persistence=0.6)
    rust_mask = np.clip((rust_n - 0.45) * 4.0, 0, 1) # sharp transition
    
    rust_tex = palette_map(fbm_noise(width, height, rng, octaves=4, persistence=0.7), [
        (0.0, (50, 15, 5)),
        (0.5, (130, 60, 20)),
        (1.0, (210, 110, 40)),
    ])
    
    b_arr = np.asarray(base, dtype=np.float32)
    r_arr = np.asarray(rust_tex, dtype=np.float32)
    m_arr = rust_mask[:, :, None]
    
    mixed = b_arr * (1 - m_arr) + r_arr * m_arr
    img = Image.fromarray(mixed.astype(np.uint8), "RGB")
    
    speck = add_speckles(width, height, rng, amount=0.005)
    out = overlay_blend(img, Image.merge("RGB", (speck, speck, speck)), opacity=0.4)
    return out.convert("RGBA")


def tex_ice(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    n = fbm_noise(width, height, rng, octaves=5, persistence=0.6)
    r = 1.0 - np.abs(2.0 * n - 1.0)
    r = r ** 3.0 # sharp glossy ridges
    
    base = palette_map(r, [
        (0.0, (180, 210, 230)),
        (0.5, (220, 240, 255)),
        (1.0, (255, 255, 255)),
    ])
    
    frost = add_speckles(width, height, rng, amount=0.01)
    out = overlay_blend(base, Image.merge("RGB", (frost, frost, frost)), opacity=0.8)
    return out.convert("RGBA")


def tex_moss(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    rock_n = fbm_noise(width, height, rng, octaves=5, persistence=0.5)
    rock = palette_map(rock_n, [(0.0, (40, 40, 40)), (1.0, (100, 100, 100))])
    
    moss_n = fbm_noise(width, height, rng, octaves=6, persistence=0.6)
    moss_mask = np.clip((moss_n - 0.4) * 3.0, 0, 1)
    
    moss_tex = palette_map(fbm_noise(width, height, rng, octaves=5, persistence=0.55), [
        (0.0, (20, 40, 10)),
        (0.5, (50, 90, 30)),
        (1.0, (90, 140, 60)),
    ])
    
    b_arr = np.asarray(rock, dtype=np.float32)
    m_arr = np.asarray(moss_tex, dtype=np.float32)
    mask = moss_mask[:, :, None]
    
    mixed = b_arr * (1 - mask) + m_arr * mask
    img = Image.fromarray(mixed.astype(np.uint8), "RGB")
    
    grain = add_grain_seeded(width, height, rng, strength=0.15)
    out = overlay_blend(img, Image.merge("RGB", (grain, grain, grain)), opacity=0.2)
    return out.convert("RGBA")


def tex_custom_photo(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """Loads a random image from CUSTOM_PHOTO_DIR and crops a width x height section from it."""
    if not CUSTOM_PHOTO_DIR or not os.path.isdir(CUSTOM_PHOTO_DIR):
        return tex_paper(width, height, rng)
        
    files = _custom_photo_files()
    
    if not files:
        return tex_paper(width, height, rng)
        
    chosen_file = rng.choice(files)
    file_path = os.path.join(CUSTOM_PHOTO_DIR, chosen_file)
    
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGBA")
            img_w, img_h = img.size
            if img_w < width or img_h < height:
                # Resize image proportionally to cover the required region
                ratio = max(width / img_w, height / img_h)
                new_w = max(width, int(img_w * ratio))
                new_h = max(height, int(img_h * ratio))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                img_w, img_h = img.size
                
            left = int(rng.integers(0, img_w - width + 1))
            top = int(rng.integers(0, img_h - height + 1))
            return img.crop((left, top, left + width, top + height))
    except Exception:
        return tex_paper(width, height, rng)


# Registry of styles
TEXTURE_STYLES: Dict[str, Callable[[int, int, np.random.Generator], Image.Image]] = {
    "paper": tex_paper,
    "parchment": tex_parchment,
    "canvas": tex_canvas,
    "concrete": tex_concrete,
    "marble": tex_marble,
    "wood": tex_wood,
    "brushed_metal": tex_brushed_metal,
    "leather": tex_leather,
    "denim": tex_denim,
    "watercolor": tex_watercolor_wash,
    "clouds": tex_clouds_sky,
    "bokeh": tex_bokeh,
    "waves": tex_waves,
    "tree_bark": tex_tree_bark,
    "sand": tex_sand,
    "camo": tex_camo,
    "rust": tex_rust,
    "ice": tex_ice,
    "moss": tex_moss,
    "custom_photo": tex_custom_photo,
}


# ============================================================
# One-call API
# ============================================================

def generate_texture_image(
    width: int,
    height: int,
    style: str = "random",
    seed: Optional[int] = None,
    add_distortion: bool = True,
    distortion_strength: float = 1.0,
    add_scribbles: bool = False,
    scribble_opacity: float = 0.10,
    add_vignette: bool = False,
) -> Tuple[Image.Image, int, str]:
    """
    Returns: (RGBA image, seed_used, style_used)

    style can be one of:
      paper, parchment, canvas, concrete, marble, wood, brushed_metal,
      leather, denim, watercolor, clouds, bokeh, or "random".
    """
    if seed is None:
        seed = secrets.randbits(32)

    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)

    if style == "random":
        random_styles = list(TEXTURE_STYLES.keys())
        if not _has_custom_photo_dir() and "custom_photo" in random_styles:
            random_styles.remove("custom_photo")
        style_used = rnd.choice(random_styles)
    else:
        if style not in TEXTURE_STYLES:
            raise ValueError(f"Unknown style '{style}'. Valid: {sorted(TEXTURE_STYLES.keys())} or 'random'")
        style_used = style

    img = TEXTURE_STYLES[style_used](width, height, rng)

    # REAL deformation
    if add_distortion:
        mag = max(2.0, min(width, height) * 0.06) * float(distortion_strength)
        grid = 5 if min(width, height) < 220 else 7
        img = mesh_warp(img, rng, grid_size=grid, magnitude=mag)

    # Optional: extra vignette on top (some styles already include one)
    if add_vignette:
        vign = radial_vignette_mask(width, height, strength=0.35, power=1.8)
        img = ImageChops.subtract(img.convert("RGB"), Image.merge("RGB", (vign, vign, vign))).convert("RGBA")

    # Optional: scribbles (usually keep OFF for realism)
    if add_scribbles:
        scrib = generate_scribble_layer(width, height, rnd)
        img = composite_with_scribbles(img, scrib, opacity=float(scribble_opacity))

    return img, seed, style_used


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    IMG_WIDTH = 350
    IMG_HEIGHT = 100

    # One random texture per run (different every time)
    img, seed, used = generate_texture_image(
        IMG_WIDTH,
        IMG_HEIGHT,
        style="random",
        seed=None,               # None => different every run
        add_distortion=True,     # real deformation
        distortion_strength=1.2, # increase for stronger warp
        add_scribbles=False,     # keep OFF for realistic textures
    )

    print("style:", used, "seed:", seed)
    img.save(f"texture_{used}_{seed}.png")  # avoids viewer caching confusion
    img.show()

    # If you want to generate a batch of different textures:
    # for i in range(9):
    #     im, sd, st = generate_texture_image(IMG_WIDTH, IMG_HEIGHT, style="random")
    #     im.save(f"batch_{i}_{st}_{sd}.png")
