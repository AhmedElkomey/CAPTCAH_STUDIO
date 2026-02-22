import os
import random
import string
import math
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
# If True, the distractor ellipses are drawn transparently (showing the background noise inside them).
# If False, the distractor ellipses are solid and block out the background noise, but still XOR the text.
TRANSPARENT_DISTRACTOR = False

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

def generate_random_string(length):
    # Ensure a mix of upper, lower and digits
    while True:
        s = ''.join(random.choices(CHARS, k=length))
        if any(c.islower() for c in s) and any(c.isupper() for c in s):
            return s

def generate_noise_layer(width, height):
    """Generates a mixture of curves, zigzags, lines, and tiny circles on a transparent layer."""
    layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    
    available_shapes = ['line', 'curve', 'zigzag', 'circle']
    # Choose a subset of shapes to use for this specific layer (1 to all)
    min_shapes = max(1, min(NOISE_SHAPES_MIN, len(available_shapes)))
    max_shapes = max(min_shapes, min(NOISE_SHAPES_MAX, len(available_shapes)))
    num_types = random.randint(min_shapes, max_shapes)
    chosen_types = random.sample(available_shapes, num_types)
    
    num_elements = random.randint(NOISE_ELEMENTS_MIN, max(NOISE_ELEMENTS_MIN, NOISE_ELEMENTS_MAX))
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
    
    distorted = np.zeros_like(img_arr)
    
    # Reduced amplitude and increased period for less chaotic scatter
    amplitude = random.uniform(2.0, 5.0) * intensity
    period = random.uniform(50.0, 150.0)
    phase = random.uniform(0, 2 * math.pi)
    
    # Decide if wave applies to full width, left half, or right half
    wave_mode = random.choice(["full", "start", "end", "middle"])
    
    for x in range(cols):
        apply_wave = False
        if wave_mode == "full":
            apply_wave = True
        elif wave_mode == "start" and x < cols * 0.5:
            apply_wave = True
        elif wave_mode == "end" and x > cols * 0.5:
            apply_wave = True
        elif wave_mode == "middle" and (cols * 0.25 < x < cols * 0.75):
            apply_wave = True
            
        if apply_wave:
            # Taper the wave effect so it doesn't break abruptly
            taper = 1.0
            if wave_mode == "start" and x > cols * 0.3:
                taper = max(0, 1.0 - (x - cols * 0.3) / (cols * 0.2))
            elif wave_mode == "end" and x < cols * 0.7:
                taper = max(0, 1.0 - (cols * 0.7 - x) / (cols * 0.2))
            elif wave_mode == "middle":
                if x < cols * 0.4:
                    taper = max(0, (x - cols * 0.25) / (cols * 0.15))
                elif x > cols * 0.6:
                    taper = max(0, 1.0 - (x - cols * 0.6) / (cols * 0.15))
                    
            y_shift = int(amplitude * taper * math.sin(2 * math.pi * x / period + phase))
        else:
            y_shift = 0
            
        for y in range(rows):
            new_y = y + y_shift
            if 0 <= new_y < rows:
                distorted[new_y, x] = img_arr[y, x]
                
    return Image.fromarray(distorted)

def apply_distractor_xor(img, mask_override=None):
    """Draws 1 or 2 random ellipses and XORs it with the background. Returns the modified image and the distractor mask."""
    width, height = img.size
    
    if mask_override is None:
        mask = create_distractor_mask(width, height)
    else:
        mask = mask_override
    
    img_arr = np.array(img.convert('RGB'))
    mask_arr = np.array(mask) > 0
    
    # Typically green-ish or bright colors
    xor_color = np.array([random.randint(50, 200), random.randint(150, 255), random.randint(50, 150)], dtype=np.uint8)
    
    blended = np.bitwise_xor(img_arr, xor_color)
    img_arr[mask_arr] = blended[mask_arr]
    
    return Image.fromarray(img_arr).convert('RGBA'), mask

def create_distractor_mask(width, height):
    """Creates a mask with 1-2 random small ellipses or circles."""
    mask = Image.new('L', (width, height), 0)
    d_draw = ImageDraw.Draw(mask)
    num_circles = random.randint(1, 2)
    for _ in range(num_circles):
        is_circle = random.choice([True, False])
        if is_circle:
            r = random.randint(15, 30) # radius, making diameter 30-60
            ellipse_w = 2 * r
            ellipse_h = 2 * r
        else:
            ellipse_w = random.randint(50, 90)  # Reduced width
            ellipse_h = random.randint(30, 60)  # Reduced height
            
        margin = 2
        x1 = random.randint(margin, max(margin, width - ellipse_w - margin))
        y1 = random.randint(margin, max(margin, height - ellipse_h - margin))
        x2 = x1 + ellipse_w
        y2 = y1 + ellipse_h
        d_draw.ellipse([x1, y1, x2, y2], fill=255)
    return mask

def create_captcha(scatter_factor=0.6, jitter_factor=0.5):
    """
    Creates a captcha with configurable destruction levels.
    """
    rand_val = random.random()
    if rand_val < RICH_BACKGROUND_PROBABILITY:
        # Ask bg_gen.py for a beautiful layered texture
        img, _, _ = generate_texture_image(
            IMG_WIDTH, 
            IMG_HEIGHT, 
            style="random", 
            add_distortion=BG_DISTORTION,
            distortion_strength=random.uniform(BG_DISTORTION_MIN, BG_DISTORTION_MAX),
            add_scribbles=BG_SCRIBBLES,
            scribble_opacity=BG_SCRIBBLE_OPACITY,
            add_vignette=BG_VIGNETTE
        )
        base_color = None
        noise_layer = generate_noise_layer(IMG_WIDTH, IMG_HEIGHT)
    elif rand_val < RICH_BACKGROUND_PROBABILITY + SOLID_BACKGROUND_PROBABILITY:
        # Completely solid color background (no noise shapes)
        base_color = (random.randint(150, 250), random.randint(150, 250), random.randint(150, 250))
        img = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), base_color)
        noise_layer = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 0))
    else:
        # 1. Base Image (Solid color)
        base_color = (random.randint(150, 250), random.randint(150, 250), random.randint(150, 250))
        img = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), base_color)
        noise_layer = generate_noise_layer(IMG_WIDTH, IMG_HEIGHT)
    
    has_distractor = random.random() > 0.2
    distractor_mask = create_distractor_mask(IMG_WIDTH, IMG_HEIGHT)
    
    # If solid distractor, cut a hole in the noise layer before compositing
    # so the noise doesn't show under the distractor
    if has_distractor and not TRANSPARENT_DISTRACTOR:
        noise_arr = np.array(noise_layer)
        mask_arr = np.array(distractor_mask)
        # Set alpha to 0 where mask is present
        noise_arr[:, :, 3] = np.where(mask_arr > 0, 0, noise_arr[:, :, 3])
        noise_layer = Image.fromarray(noise_arr)

    # Combine background and noise_layer
    img = Image.alpha_composite(img, noise_layer)

    # 2. Add Text
    adj_min = min(MIN_CHARS, MAX_CHARS)
    adj_max = max(MIN_CHARS, MAX_CHARS)
    text_length = random.randint(adj_min, adj_max)
    text = generate_random_string(text_length)
    text_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
    
    # Pick a random font for the ENTIRE word
    font_path = random.choice(AVAILABLE_FONTS) if AVAILABLE_FONTS else "arial.ttf"
    try:
        font = ImageFont.truetype(font_path, size=random.randint(50, 65))
    except IOError:
        font = ImageFont.load_default()
        
    # Calculate starting position
    char_w_estimate = int(font.getbbox('A')[2] if hasattr(font, 'getbbox') else font.getsize('A')[0])
    advance_estimate = int(char_w_estimate * (0.55 + 0.25 * scatter_factor))
    estimated_total_width = text_length * advance_estimate
    
    current_x = max(10, (IMG_WIDTH - estimated_total_width) // 2) + random.randint(-10, 10)
    
    for char in text:
        bbox = font.getbbox(char) if hasattr(font, 'getbbox') else font.getsize(char)
        if hasattr(font, 'getbbox'):
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
        else:
            char_w, char_h = bbox
            
        c_img = Image.new('RGBA', (char_w * 3, char_h * 3), (255, 255, 255, 0))
        c_draw = ImageDraw.Draw(c_img)
        
        t_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), 255)
        
        c_draw.text((char_w, char_h), char, font=font, fill=t_color)
        
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
        base_y = (IMG_HEIGHT - c_img.height) // 2
        paste_y = base_y + y_jitter
        
        text_layer.paste(c_img, (current_x, paste_y), c_img)
        
        advance = int(c_img.width * random.uniform(0.7, 1 + 0.3 * scatter_factor))
        current_x += advance

    # 3. Apply RANDOM Wave Distortion to Text Layer
    if random.random() > 0.4:
        text_layer = apply_wave_distortion(text_layer, intensity=scatter_factor)

    # Combine background and text
    img = Image.alpha_composite(img, text_layer)
    
    # 4. Add XOR Distractor over EVERYTHING using the generated mask
    if has_distractor:
        img, _ = apply_distractor_xor(img, mask_override=distractor_mask)
        
    return img.convert("RGB"), text
        
    return img.convert("RGB"), text

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating {NUM_IMAGES} synthetic CAPTCHAs...")
    print(f"TRANSPARENT_DISTRACTOR is set to: {TRANSPARENT_DISTRACTOR}")
    
    for i in range(NUM_IMAGES):
        img, label = create_captcha()
        
        idx = f"{i:04d}"
        filename = f"captcha_v5_{idx}_{label}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        img.save(filepath, quality=95)
        print(f"Saved {filepath}")

if __name__ == "__main__":
    main()
