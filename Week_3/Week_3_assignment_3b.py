import numpy as np
from PIL import Image

def expand_image(img, pad=2):
    h, w, c = img.shape
    expanded = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=img.dtype)
    expanded[pad:pad + h, pad:pad + w] = img
    expanded[:pad, pad:pad + w] = img[0, :]
    expanded[pad + h:, pad:pad + w] = img[-1, :]
    expanded[:, :pad] = expanded[:, pad:pad + 1]
    expanded[:, pad + w:] = expanded[:, pad + w - 1:pad + w]
    return expanded

def convolve_image(img, kernel):
    kh, kw = kernel.shape
    pad = kh // 2  # assumes square, odd-sized kernel (5x5 -> pad=2)

    h, w, c = img.shape
    padded = expand_image(img.astype(np.float64), pad=pad)

    output = np.zeros((h, w, c), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                region = padded[y:y + kh, x:x + kw, ch]
                output[y, x, ch] = np.sum(region * kernel)

    # Clip to valid pixel range and convert back to uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

# --- Gaussian 5x5 kernel from the assignment ---
gaussian_kernel = np.array([
    [0.00390625, 0.015625,   0.0234375, 0.015625,   0.00390625],
    [0.015625,   0.0625,     0.09375,   0.0625,     0.015625  ],
    [0.0234375,  0.09375,    0.140625,  0.09375,    0.0234375 ],
    [0.015625,   0.0625,     0.09375,   0.0625,     0.015625  ],
    [0.00390625, 0.015625,   0.0234375, 0.015625,   0.00390625],
])
print("Kernel sums to:", gaussian_kernel.sum())  # should be ~1.0
# --- Load, filter, and save ---
img = np.array(Image.open("shirt.jpg").convert("RGB"))
blurred = convolve_image(img, gaussian_kernel)
Image.fromarray(img).save("shirt_original.png")
Image.fromarray(blurred).save("shirt_blurred.png")

def unsharp_mask(original, blurred):
    return original.astype(np.int16) - blurred.astype(np.int16)

def sharpen(original, mask, multiplier=1.0):

    sharpened = original.astype(np.float64) + multiplier * mask.astype(np.float64)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# Usage
original = np.array(Image.open("17.png").convert("RGB"))
blurred = convolve_image(original, gaussian_kernel)   # from 3b
Image.fromarray(blurred).save("17_blurred.png")
mask = unsharp_mask(original, blurred)
sharpened = sharpen(original, mask, multiplier=1.0)
Image.fromarray(sharpened).save("17_sharpened.png")

def resize_nearest(img, scale):
    img = np.array(Image.open("shirt.jpg").convert("RGB"))
    h, w, c = img.shape
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    row_idx = np.clip((np.arange(new_h) / scale).astype(int), 0, h - 1)
    col_idx = np.clip((np.arange(new_w) / scale).astype(int), 0, w - 1)

    return img[row_idx][:, col_idx]


SCALE = 0.17

# Without anti-aliasing: scale directly
resized_no_filter = resize_nearest(original, SCALE)
Image.fromarray(resized_no_filter).save("shirt_resized_no_filter.png")

# With anti-aliasing: blur first (5x5 Gaussian function)
blurred = convolve_image(original, gaussian_kernel)
resized_anti_aliased = resize_nearest(blurred, SCALE)
Image.fromarray(resized_anti_aliased).save("shirt_resized_anti_aliased.png")