def create_patched_image(image, x, y, size):
    patched = image.copy()
    half = size // 2
    x1, x2 = max(0, x - half), min(299, x + half)           # keep the box inside the image
    y1, y2 = max(0, y - half), min(299, y + half)
    patched[y1:y2, x1:x2] = 128                             # paint a gray square
    return patched
