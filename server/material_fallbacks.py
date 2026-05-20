from PIL import Image, ImageDraw
import numpy as np
import os


def texture_has_visible_detail(image: Image.Image, min_std: float = 6.0) -> bool:
    """Return False for near-flat textures that read as a single color."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return False
    luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(luminance.std()) >= min_std


def make_procedural_room_texture(surface_type: str, size: int = 1024) -> Image.Image:
    """Create a deterministic, high-readability fallback texture for room surfaces."""
    surface_type = surface_type.lower()
    rng = np.random.default_rng(17 if surface_type == "floor" else 31)

    if surface_type == "floor":
        base = np.zeros((size, size, 3), dtype=np.uint8)
        base[:] = np.array([156, 126, 88], dtype=np.uint8)
        noise = rng.normal(0, 9, base.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(base, "RGB")
        draw = ImageDraw.Draw(image)

        plank_h = max(54, size // 14)
        for y in range(0, size, plank_h):
            draw.line([(0, y), (size, y)], fill=(88, 67, 43), width=max(3, size // 256))
            offset = (y // plank_h % 3) * size // 5
            for x in range(-offset, size, size // 3):
                draw.line([(x, y), (x, min(size, y + plank_h))], fill=(105, 80, 50), width=max(2, size // 384))
        return image

    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[:] = np.array([218, 213, 202], dtype=np.uint8)
    noise = rng.normal(0, 5, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(image)

    stripe_w = max(80, size // 10)
    for x in range(0, size, stripe_w):
        color = (198, 193, 184) if (x // stripe_w) % 2 else (232, 228, 219)
        draw.rectangle([x, 0, min(size, x + stripe_w // 3), size], fill=color)
    for y in range(0, size, max(90, size // 12)):
        draw.line([(0, y), (size, y)], fill=(188, 183, 174), width=max(1, size // 512))
    return image


def ensure_visible_room_texture(image: Image.Image | None, surface_type: str) -> Image.Image:
    if image is None or not texture_has_visible_detail(image):
        return make_procedural_room_texture(surface_type)
    return image.convert("RGB")


def ensure_material_file(texture_path: str, surface_type: str) -> str:
    os.makedirs(os.path.dirname(texture_path), exist_ok=True)
    if os.path.exists(texture_path):
        return texture_path
    make_procedural_room_texture(surface_type).save(texture_path)
    return texture_path
