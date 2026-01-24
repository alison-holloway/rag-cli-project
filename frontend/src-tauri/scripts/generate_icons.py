#!/usr/bin/env python3
"""
Generate app icons for RAG Assistant.
Creates all required sizes for macOS, Windows, and Linux.
"""

import os
import subprocess
from PIL import Image, ImageDraw, ImageFont
import math

# Icon sizes needed for different platforms
ICON_SIZES = {
    # macOS
    "icon.png": 1024,
    "128x128.png": 128,
    "128x128@2x.png": 256,
    "32x32.png": 32,
    # Windows
    "Square30x30Logo.png": 30,
    "Square44x44Logo.png": 44,
    "Square71x71Logo.png": 71,
    "Square89x89Logo.png": 89,
    "Square107x107Logo.png": 107,
    "Square142x142Logo.png": 142,
    "Square150x150Logo.png": 150,
    "Square284x284Logo.png": 284,
    "Square310x310Logo.png": 310,
    "StoreLogo.png": 50,
}


def create_gradient(width, height, color1, color2):
    """Create a diagonal gradient image."""
    img = Image.new("RGBA", (width, height))
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            # Diagonal gradient
            ratio = (x + y) / (width + height)
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            pixels[x, y] = (r, g, b, 255)

    return img


def create_rounded_mask(size, radius):
    """Create a rounded rectangle mask."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (size - 1, size - 1)], radius=radius, fill=255)
    return mask


def create_icon(size, master_icon=None):
    """Create a single icon at the specified size.

    If master_icon is provided, resize from that instead of creating new.
    This ensures consistent appearance across all sizes.
    """
    if master_icon is not None:
        # Resize from master using high-quality resampling
        return master_icon.resize((size, size), Image.Resampling.LANCZOS)

    # Colors for gradient (purple to indigo)
    color1 = (102, 126, 234)  # #667eea
    color2 = (118, 75, 162)   # #764ba2

    # Create gradient background
    img = create_gradient(size, size, color1, color2)

    # Apply rounded corners (about 22% radius like macOS)
    radius = int(size * 0.22)
    mask = create_rounded_mask(size, radius)

    # Create final image with transparency
    final = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    final.paste(img, mask=mask)

    # Draw the "R" letter
    draw = ImageDraw.Draw(final)

    # Calculate font size (about 45% of icon size)
    font_size = max(10, int(size * 0.45))

    # Try to use a system font, fall back to default
    font = None
    font_paths = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue

    if font is None:
        try:
            # Try to use a basic truetype font
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
        except Exception:
            # Last resort - use default
            font = ImageFont.load_default()

    # Calculate text position (centered)
    text = "R"
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) // 2 - bbox[0]
        y = (size - text_height) // 2 - bbox[1]
    except Exception:
        # Fallback for tiny sizes
        x = size // 4
        y = size // 4

    # Draw text with slight shadow for depth
    shadow_offset = max(1, size // 100)
    try:
        draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0, 50), font=font)
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    except Exception:
        # Skip text for very small icons
        pass

    # Add subtle highlight at top
    highlight = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight)

    # Create a subtle top-to-bottom gradient overlay
    highlight_height = max(1, size // 4)
    for i in range(highlight_height):
        alpha = int(30 * (1 - i / highlight_height))
        highlight_draw.line([(0, i), (size, i)], fill=(255, 255, 255, alpha))

    final = Image.alpha_composite(final, highlight)

    return final


def create_icns(icons_dir, master_icon):
    """Create .icns file for macOS using iconutil."""
    iconset_dir = os.path.join(icons_dir, "AppIcon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)

    # macOS iconset sizes
    iconset_sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }

    for filename, size in iconset_sizes.items():
        icon = create_icon(size, master_icon)
        icon.save(os.path.join(iconset_dir, filename), "PNG")

    # Use iconutil to create .icns
    icns_path = os.path.join(icons_dir, "icon.icns")
    try:
        subprocess.run(
            ["iconutil", "-c", "icns", iconset_dir, "-o", icns_path],
            check=True,
            capture_output=True,
        )
        print(f"Created {icns_path}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not create .icns file: {e}")
    except FileNotFoundError:
        print("Warning: iconutil not found, skipping .icns creation")

    # Clean up iconset directory
    import shutil
    shutil.rmtree(iconset_dir, ignore_errors=True)


def create_ico(icons_dir, master_icon):
    """Create .ico file for Windows."""
    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [create_icon(size, master_icon) for size in ico_sizes]

    ico_path = os.path.join(icons_dir, "icon.ico")
    images[0].save(
        ico_path,
        format="ICO",
        sizes=[(img.width, img.height) for img in images],
        append_images=images[1:],
    )
    print(f"Created {ico_path}")


def main():
    # Get the icons directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "..", "icons")
    os.makedirs(icons_dir, exist_ok=True)

    print(f"Generating icons in {icons_dir}")

    # Create master icon at 1024x1024 first
    print("Creating master icon (1024x1024)...")
    master_icon = create_icon(1024)

    # Create standard PNG icons by resizing from master
    for filename, size in ICON_SIZES.items():
        icon = create_icon(size, master_icon)
        filepath = os.path.join(icons_dir, filename)
        icon.save(filepath, "PNG")
        print(f"Created {filename} ({size}x{size})")

    # Create .icns for macOS
    create_icns(icons_dir, master_icon)

    # Create .ico for Windows
    create_ico(icons_dir, master_icon)

    print("\nIcon generation complete!")


if __name__ == "__main__":
    main()
