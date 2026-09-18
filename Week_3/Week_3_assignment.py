import numpy as np

def expanded_image(img):
    h, w, c = img.shape
    # Create expanded image
    expanded = np.zeros((h + 4, w + 4, c), dtype=img.dtype)
    # Copy original image into center
    expanded[2:h+2, 2:w+2] = img
    # Copy top edge
    expanded[:2, 2:w+2] = img[0:1, :, :]
    # Copy bottom edge
    expanded[h+2:, 2:w+2] = img[-1:, :, :]
    # Copy left edge
    expanded[:, :2] = expanded[:, 2:3]
    # Copy right edge
    expanded[:, w+2:] = expanded[:, w+1:w+2]
    return expanded

# random 5x5x3 image
np.random.seed(42)
img_random = np.random.randint(0, 256, (5, 5, 3))
expanded_random = expanded_image(img_random)

print("Random RGB Image")
print("Original shape:", img_random.shape)
print("Original dimensions:", img_random.ndim)
print(img_random)
print("\nExpanded shape:", expanded_random.shape)
print("Expanded dimensions:", expanded_random.ndim)
print(expanded_random)

# demonstration
img_demo = np.zeros((5, 5, 3), dtype=int)
counter = 1
for y in range(5):
    for x in range(5):
        img_demo[y, x, 0] = counter
        counter += 1

expanded_demo = expanded_image(img_demo)
print("\n\nDemo using numbered pixels")
print("\nOriginal (channel 0):")
print(img_demo[:, :, 0])
print("\nExpanded (channel 0):")
print(expanded_demo[:, :, 0])