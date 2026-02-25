import os
import random
import string
import math
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from bg_gen import generate_texture_image

# Default settings
OUTPUT_DIR = "generated_captchas"
NUM_IMAGES = 10
IMG_WIDTH = 350
IMG_HEIGHT = 100
CHARS = string.ascii_letters + string.digits
MIN_CHARS = 6
MAX_CHARS = 8

# --- CONFIGURATION FLAGS ---
CUSTOM_PHOTO_DIR = None

# If > 0, specifies probability of using rich backgrounds from bg_gen.py
# If 0, uses the standard solid color with basic noise shapes
RICH_BACKGROUND_PROBABILITY = 1.0

# If > 0, specifies probability of a completely blank solid color background (no noise)
SOLID_BACKGROUND_PROBABILITY = 0.0

# Rich background tuning
BG_DISTORTION = True
BG_DISTORTION_MIN = 0.8
BG_DISTORTION_MAX = 1.5
BG_SCRIBBLES = False
BG_SCRIBBLE_OPACITY = 0.10
BG_VIGNETTE = False

# Noise layer config
NOISE_SHAPES_MIN = 1
NOISE_SHAPES_MAX = 4
NOISE_ELEMENTS_MIN = 40
NOISE_ELEMENTS_MAX = 60

# A mix of common Windows fonts (Normal and Bold)
FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf",
    "C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/courbd.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/trebuc.ttf", "C:/Windows/Fonts/trebucbd.ttf",
    "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/comic.ttf", "C:/Windows/Fonts/comicbd.ttf",
    "C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/pala.ttf", "C:/Windows/Fonts/palab.ttf",
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/lucon.ttf",
    "C:/Windows/Fonts/framd.ttf", "C:/Windows/Fonts/framdit.ttf"
]
# Filter out fonts that might not exist on the user's system just in case
AVAILABLE_FONTS = [f for f in FONT_PATHS if os.path.exists(f)]


def build_runtime_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Builds a concrete runtime config so preview and batch use the same settings."""
    config: Dict[str, Any] = {
        "img_width": IMG_WIDTH,
        "img_height": IMG_HEIGHT,
        "min_chars": MIN_CHARS,
        "max_chars": MAX_CHARS,
        "rich_background_probability": RICH_BACKGROUND_PROBABILITY,
        "solid_background_probability": SOLID_BACKGROUND_PROBABILITY,
        "bg_distortion": BG_DISTORTION,
        "bg_distortion_min": BG_DISTORTION_MIN,
        "bg_distortion_max": BG_DISTORTION_MAX,
        "bg_scribbles": BG_SCRIBBLES,
        "bg_scribble_opacity": BG_SCRIBBLE_OPACITY,
        "bg_vignette": BG_VIGNETTE,
        "noise_shapes_min": NOISE_SHAPES_MIN,
        "noise_shapes_max": NOISE_SHAPES_MAX,
        "noise_elements_min": NOISE_ELEMENTS_MIN,
        "noise_elements_max": NOISE_ELEMENTS_MAX,
        "available_fonts": list(AVAILABLE_FONTS),
        "custom_photo_dir": CUSTOM_PHOTO_DIR,
    }
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    return config


def generate_random_string(length):
    # Ensure a mix of upper, lower and digits
    while True:
        s = ''.join(random.choices(CHARS, k=length))
        if any(c.islower() for c in s) and any(c.isupper() for c in s):
            return s

def generate_noise_layer(width, height, shapes_min=None, shapes_max=None, elements_min=None, elements_max=None):
    """Generates a mixture of curves, zigzags, lines, and tiny circles on a transparent layer."""
    layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    shape_min = NOISE_SHAPES_MIN if shapes_min is None else shapes_min
    shape_max = NOISE_SHAPES_MAX if shapes_max is None else shapes_max
    element_min = NOISE_ELEMENTS_MIN if elements_min is None else elements_min
    element_max = NOISE_ELEMENTS_MAX if elements_max is None else elements_max
    
    available_shapes = ['line', 'curve', 'zigzag', 'circle']
    # Choose a subset of shapes to use for this specific layer (1 to all)
    min_shapes = max(1, min(shape_min, len(available_shapes)))
    max_shapes = max(min_shapes, min(shape_max, len(available_shapes)))
    num_types = random.randint(min_shapes, max_shapes)
    chosen_types = random.sample(available_shapes, num_types)
    
    num_elements = random.randint(element_min, max(element_min, element_max))
    for _ in range(num_elements):
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)
        shape_type = random.choice(chosen_types)
        
        if shape_type == 'line':
            x1 = random.randint(-50, width+50)
            y1 = random.randint(-50, height+50)
            x2 = random.randint(-50, width+50)
            y2 = random.randint(-50, height+50)
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(2, 4))
            
        elif shape_type == 'curve':
            x1 = random.randint(-50, width)
            y1 = random.randint(-50, height)
            x2 = random.randint(x1 + 10, width + 50)
            y2 = random.randint(y1 + 10, height + 50)
            bbox = [x1, y1, x2, y2]
            start_angle = random.randint(0, 180)
            end_angle = start_angle + random.randint(45, 180)
            draw.arc(bbox, start=start_angle, end=end_angle, fill=color, width=random.randint(2, 4))
            
        elif shape_type == 'zigzag':
            points = []
            x, y = random.randint(0, width), random.randint(0, height)
            points.append((x, y))
            for _ in range(random.randint(2, 4)):
                x += random.randint(-30, 30)
                y += random.randint(-30, 30)
                points.append((x, y))
            # PIL doesn't have joint param in standard line, so we just draw lines
            draw.line(points, fill=color, width=random.randint(1, 3))
            
        elif shape_type == 'circle':
            r = random.randint(3, 12)
            cx = random.randint(0, width)
            cy = random.randint(0, height)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=random.randint(1, 4))
            
    return layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

def apply_wave_distortion(img, intensity=1.0):
    """Applies a RANDOMIZED partial or full sinusoidal wave distortion to an image."""
    img_arr = np.array(img)
    if img_arr.shape[2] == 4:
        img_arr = img_arr.copy()
    rows, cols = img_arr.shape[:2]
    
    # Reduced amplitude and increased period for less chaotic scatter
    amplitude = random.uniform(2.0, 5.0) * intensity
    period = random.uniform(50.0, 150.0)
    phase = random.uniform(0, 2 * math.pi)
    
    # Decide if wave applies to full width, left half, or right half
    wave_mode = random.choice(["full", "start", "end", "middle"])

    x = np.arange(cols, dtype=np.float32)
    apply_wave = np.zeros(cols, dtype=bool)
    if wave_mode == "full":
        apply_wave[:] = True
    elif wave_mode == "start":
        apply_wave = x < cols * 0.5
    elif wave_mode == "end":
        apply_wave = x > cols * 0.5
    elif wave_mode == "middle":
        apply_wave = (x > cols * 0.25) & (x < cols * 0.75)

    taper = np.ones(cols, dtype=np.float32)
    if wave_mode == "start":
        region = x > cols * 0.3
        taper[region] = np.maximum(0.0, 1.0 - (x[region] - cols * 0.3) / (cols * 0.2))
    elif wave_mode == "end":
        region = x < cols * 0.7
        taper[region] = np.maximum(0.0, 1.0 - (cols * 0.7 - x[region]) / (cols * 0.2))
    elif wave_mode == "middle":
        left = x < cols * 0.4
        right = x > cols * 0.6
        taper[left] = np.maximum(0.0, (x[left] - cols * 0.25) / (cols * 0.15))
        taper[right] = np.maximum(0.0, 1.0 - (x[right] - cols * 0.6) / (cols * 0.15))

    y_shift = np.zeros(cols, dtype=np.int32)
    y_shift[apply_wave] = (
        amplitude
        * taper[apply_wave]
        * np.sin(2 * math.pi * x[apply_wave] / period + phase)
    ).astype(np.int32)

    distorted = np.zeros_like(img_arr)
    src_y = np.broadcast_to(np.arange(rows, dtype=np.int32)[:, None], (rows, cols))
    src_x = np.broadcast_to(np.arange(cols, dtype=np.int32)[None, :], (rows, cols))
    dst_y = src_y + y_shift[None, :]
    valid = (dst_y >= 0) & (dst_y < rows)
    distorted[dst_y[valid], src_x[valid]] = img_arr[src_y[valid], src_x[valid]]
                
    return Image.fromarray(distorted)

def apply_two_blobs(combined_img, text_layer):
    """
    Applies the adversarial 'Two Blobs' rule.
    - Massive solid opaque ellipses in the center, overlapping text.
    - Offset positions.
    - Dark high-contrast or neon colors.
    - Opaque over background, text colour-shifted where blobs overlap text.
    """
    width, height = combined_img.size

    num_blobs = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
    if num_blobs == 0:
        return combined_img

    # Supersample: Draw at 4x resolution then downscale for perfect, clean anti-aliasing without glow.
    scale = 4
    blobs_layer_large = Image.new('RGBA', (width * scale, height * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(blobs_layer_large)

    # Choose color
    if random.random() > 0.8:
        # Neon
        colors = [(57, 255, 20), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
    else:
        # Dark high-contrast
        colors = [(0, 0, 0), (20, 20, 20), (0, 40, 0), (40, 0, 0), (0, 0, 40)]

    base_color = random.choice(colors)
    color1 = (*base_color, 255)

    # Size constraints
    h1 = int(random.uniform(0.3, 0.5) * height)
    w1 = int(random.uniform(1.5, 2.0) * h1)

    # Placement
    center_x = width // 2
    center_y = height // 2

    # Left blob
    x1 = center_x - w1 + random.randint(0, int(w1 * 0.4))
    y1 = center_y - (h1 // 2) + random.randint(-15, 15)

    d.ellipse([x1 * scale, y1 * scale, (x1 + w1) * scale, (y1 + h1) * scale], fill=color1)

    if num_blobs == 2:
        # Slight variation for the second blob
        c_offset = random.randint(-15, 15)
        color2 = (
            max(0, min(255, base_color[0] + c_offset)),
            max(0, min(255, base_color[1] + c_offset)),
            max(0, min(255, base_color[2] + c_offset)),
            255
        )
        h2 = int(random.uniform(0.3, 0.5) * height)
        w2 = int(random.uniform(1.5, 2.0) * h2)

        # Right blob
        x2 = center_x - random.randint(0, int(w2 * 0.4))
        y2 = center_y - (h2 // 2) + random.randint(-15, 15)

        # Force Y offset rule (5-15px different)
        if abs(y1 - y2) < 5:
            y2 += random.choice([-1, 1]) * random.randint(5, 15)

        d.ellipse([x2 * scale, y2 * scale, (x2 + w2) * scale, (y2 + h2) * scale], fill=color2)

    # Downscale using high-quality Lanczos resampling to get smooth edges.
    resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
    blobs_layer = blobs_layer_large.resize((width, height), resample=resample_filter)

    arr_combined = np.array(combined_img).astype(np.float32)
    arr_text = np.array(text_layer).astype(np.float32)
    arr_blob = np.array(blobs_layer).astype(np.float32)

    # Normalised alpha channels (0.0 – 1.0), expanded for broadcasting.
    blob_alpha = (arr_blob[:, :, 3] / 255.0)[:, :, np.newaxis]
    text_alpha = (arr_text[:, :, 3] / 255.0)[:, :, np.newaxis]

    # Where blobs have any opacity (after blur, edges are fractional).
    blob_mask = arr_blob[:, :, 3] > 2  # ignore near-zero fringe

    out = arr_combined.copy()

    # Inverted text colour (used where text overlaps blobs).
    inv_text_rgb = 255.0 - arr_text[:, :, :3]

    # Blend: blob base colour mixed with inverted text weighted by text alpha.
    blended_rgb = arr_blob[:, :, :3] * (1.0 - text_alpha) + inv_text_rgb * text_alpha

    # Composite the blended result over the existing background by blob alpha.
    composited_rgb = out[:, :, :3] * (1.0 - blob_alpha) + blended_rgb * blob_alpha

    # Apply only where blob is present.
    mask3 = blob_mask[:, :, np.newaxis].repeat(3, axis=2)
    out[:, :, :3] = np.where(mask3, composited_rgb, out[:, :, :3])
    out[:, :, 3] = np.where(blob_mask,
                             np.minimum(255.0, out[:, :, 3] + blob_alpha[:, :, 0] * 255.0),
                             out[:, :, 3])

    return Image.fromarray(out.astype(np.uint8))


def create_captcha(scatter_factor=0.6, jitter_factor=0.5, texture_style="random", config=None):
    """
    Creates a captcha with configurable destruction levels.
    """
    cfg = build_runtime_config(config)
    img_width = int(cfg["img_width"])
    img_height = int(cfg["img_height"])
    min_chars = int(cfg["min_chars"])
    max_chars = int(cfg["max_chars"])
    rich_background_probability = max(0.0, min(1.0, float(cfg["rich_background_probability"])))
    solid_background_probability = max(0.0, min(1.0, float(cfg["solid_background_probability"])))
    bg_distortion = bool(cfg["bg_distortion"])
    bg_distortion_min = float(cfg["bg_distortion_min"])
    bg_distortion_max = float(cfg["bg_distortion_max"])
    bg_scribbles = bool(cfg["bg_scribbles"])
    bg_scribble_opacity = float(cfg["bg_scribble_opacity"])
    bg_vignette = bool(cfg["bg_vignette"])
    noise_shapes_min = int(cfg["noise_shapes_min"])
    noise_shapes_max = int(cfg["noise_shapes_max"])
    noise_elements_min = int(cfg["noise_elements_min"])
    noise_elements_max = int(cfg["noise_elements_max"])
    available_fonts = list(cfg.get("available_fonts", []))

    import bg_gen
    bg_gen.CUSTOM_PHOTO_DIR = cfg.get("custom_photo_dir")

    # If a concrete style is selected, always use the rich texture path.
    if texture_style != "random":
        rich_background_probability = 1.0
        solid_background_probability = 0.0
    
    rand_val = random.random()
    if rand_val < rich_background_probability:
        # Ask bg_gen.py for a beautiful layered texture
        img, _, style_used = generate_texture_image(
            img_width, 
            img_height, 
            style=texture_style, 
            add_distortion=bg_distortion,
            distortion_strength=random.uniform(bg_distortion_min, bg_distortion_max),
            add_scribbles=bg_scribbles,
            scribble_opacity=bg_scribble_opacity,
            add_vignette=bg_vignette
        )
        base_color = None
        if style_used == "custom_photo":
            # Keep custom photo backgrounds clean (no synthetic shape noise).
            noise_layer = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        else:
            noise_layer = generate_noise_layer(
                img_width,
                img_height,
                shapes_min=noise_shapes_min,
                shapes_max=noise_shapes_max,
                elements_min=noise_elements_min,
                elements_max=noise_elements_max,
            )
    elif rand_val < rich_background_probability + solid_background_probability:
        # Completely solid color background (no noise shapes)
        base_color = (random.randint(150, 250), random.randint(150, 250), random.randint(150, 250))
        img = Image.new('RGBA', (img_width, img_height), base_color)
        noise_layer = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    else:
        # 1. Base Image (Solid color)
        base_color = (random.randint(150, 250), random.randint(150, 250), random.randint(150, 250))
        img = Image.new('RGBA', (img_width, img_height), base_color)
        noise_layer = generate_noise_layer(
            img_width,
            img_height,
            shapes_min=noise_shapes_min,
            shapes_max=noise_shapes_max,
            elements_min=noise_elements_min,
            elements_max=noise_elements_max,
        )
    
    # Combine background and noise_layer
    img = Image.alpha_composite(img, noise_layer)

    # 2. Add Text
    adj_min = min(min_chars, max_chars)
    adj_max = max(min_chars, max_chars)
    text_length = random.randint(adj_min, adj_max)
    text = generate_random_string(text_length)
    text_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    
    # Pick a random font for the ENTIRE word
    if available_fonts:
        font_path = random.choice(available_fonts)
    else:
        font_path = "arial.ttf"
    try:
        font = ImageFont.truetype(font_path, size=random.randint(50, 65))
    except IOError:
        font = ImageFont.load_default()
        
    # Calculate starting position
    char_w_estimate = int(font.getbbox('A')[2] if hasattr(font, 'getbbox') else font.getsize('A')[0])
    advance_estimate = int(char_w_estimate * (0.55 + 0.25 * scatter_factor))
    estimated_total_width = text_length * advance_estimate
    
    current_x = max(10, (img_width - estimated_total_width) // 2) + random.randint(-10, 10)
    # Keep one random text color per captcha image.
    text_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), 255)
    
    for char in text:
        bbox = font.getbbox(char) if hasattr(font, 'getbbox') else font.getsize(char)
        if hasattr(font, 'getbbox'):
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
        else:
            char_w, char_h = bbox
            
        c_img = Image.new('RGBA', (char_w * 3, char_h * 3), (0, 0, 0, 0))
        c_draw = ImageDraw.Draw(c_img)

        c_draw.text((char_w, char_h), char, font=font, fill=text_color)
        
        # Reduced angle for less scatter
        max_angle = 25 * scatter_factor
        angle = random.uniform(-max_angle, max_angle)
        c_img = c_img.rotate(angle, resample=Image.BICUBIC, expand=1, fillcolor=(0, 0, 0, 0))
        
        # Crop tight to allow overlapping
        c_bbox = c_img.getbbox()
        if c_bbox:
            c_img = c_img.crop(c_bbox)
            
        # Reduced vertical jitter
        max_jitter = int(5 * jitter_factor)
        y_jitter = random.randint(-max_jitter, max_jitter)
        base_y = (img_height - c_img.height) // 2
        paste_y = base_y + y_jitter
        
        text_layer.paste(c_img, (current_x, paste_y), c_img)
        
        advance = int(c_img.width * random.uniform(0.7, 1 + 0.3 * scatter_factor))
        current_x += advance

    # 3. Apply RANDOM Wave Distortion to Text Layer
    if random.random() > 0.4:
        text_layer = apply_wave_distortion(text_layer, intensity=scatter_factor)

    # Combine background and text
    img = Image.alpha_composite(img, text_layer)

    # 4. Apply blobs and text blend – pass the (possibly distorted) text_layer
    #    so the intersection mask matches the actual character positions.
    img = apply_two_blobs(img, text_layer)
        
    return img.convert("RGB"), text

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating {NUM_IMAGES} synthetic CAPTCHAs...")
    
    for i in range(NUM_IMAGES):
        img, label = create_captcha()
        
        idx = f"{i:04d}"
        filename = f"captcha_v5_{idx}_{label}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        img.save(filepath, quality=95)
        print(f"Saved {filepath}")

if __name__ == "__main__":
    main()
