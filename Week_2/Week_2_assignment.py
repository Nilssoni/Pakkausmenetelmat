from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#print("NumPy version:", np.__version__)
#print("PIL version:", Image.__version__)
#print("Matplotlib version:", matplotlib.__version__)
#print("Visual studio code version:", "1.79.2")

def size_and_color(image_path):
    picture=Image.open(image_path)

    width, height = picture.size
    print(f"Image {picture.filename} size: {width} x {height}px")

    pixels=np.array(picture)
    # Height, Width, Colors (RGB)
    red = pixels[0, 0, 0]
    green = pixels[0, 0, 1]
    blue = pixels[0, 0, 2]

    print(f"Top-left corner RGB values: Red={red}, Green={green}, Blue={blue}")

def scaling(image_path, scale_factor):
    image = Image.open(image_path)
    pixels_old = np.array(image)

    height_old, width_old, channels = pixels_old.shape

    width_new = int(width_old * scale_factor)
    height_new = int(height_old * scale_factor)

    pixels_new = np.zeros((height_new, width_new, channels), dtype=np.uint8)

    for y_new in range(height_new):
        for x_new in range(width_new):
            x_old = int(round(x_new / scale_factor))
            y_old = int(round(y_new / scale_factor))

            x_old= min(x_old, width_old - 1)
            y_old = min(y_old, height_old - 1)

            pixels_new[y_new, x_new] = pixels_old[y_old, x_old]

    scaled_image = Image.fromarray(pixels_new)
    print(f"Scaled image size: {width_new} x {height_new}px")
    scaled_image.save(f"scaled_shirt_x4.png")
    return scaled_image

def scaling_bilinear(image_path, scale_factor):
    image = Image.open(image_path)
    pixels_old = np.array(image, dtype=np.float32)

    height_old, width_old, channels = pixels_old.shape

    width_new = int(width_old * scale_factor)
    height_new = int(height_old * scale_factor)

    pixels_new = np.zeros((height_new, width_new, channels), dtype=np.float32)

    for y_new in range(height_new):
        for x_new in range(width_new):
            x_old_float = x_new / scale_factor
            y_old_float = y_new / scale_factor

            x1 = int(np.floor(x_old_float))
            y1 = int(np.floor(y_old_float))

            x2 = min(x1 + 1, width_old - 1)
            y2 = min(y1 + 1, height_old - 1)

            dx = x_old_float - x1
            dy = y_old_float - y1

            q11 = pixels_old[y1, x1]
            q21 = pixels_old[y1, x2]
            q12 = pixels_old[y2, x1]
            q22 = pixels_old[y2, x2]

            interpolated_pixels = (
                (1 - dx) * (1 - dy) * q11
                + dx * (1 - dy) * q21
                + (1 - dx) * dy * q12
                + dx * dy * q22
            )

            pixels_new[y_new, x_new] = interpolated_pixels

    pixels_new = np.clip(pixels_new, 0, 255).astype(np.uint8)
    scaled_image = Image.fromarray(pixels_new)

    print(f"Bilinear scaled image size: {width_new} x {height_new}px")
    scaled_image.save("shirt_bilinear_x4.png")
    return scaled_image

def grayscale(image_path):
    image = Image.open(image_path)
    pixels = np.array(image, dtype=np.float32)

    r = pixels[:, :, 0]
    g = pixels[:, :, 1]
    b = pixels[:, :, 2]

    gray_avg = (r + g + b) / 3
    gray_pixels = np.zeros_like(pixels, dtype=np.uint8)
    gray_pixels[:, :, 0] = gray_avg
    gray_pixels[:, :, 1] = gray_avg
    gray_pixels[:, :, 2] = gray_avg

    gray_image = Image.fromarray(gray_pixels)
    gray_image.save(f"grayscale_{image_path}")
    gray_image.show()

def color_quantization(image_path, target_bits):
    image = Image.open(image_path)
    pixels = np.array(image, dtype=np.uint8)

    shift = 8 - target_bits
    quantized_pixels = (pixels >> shift) << shift

    output_filename = f"linnanmaa_{target_bits}bits.png"
    result_image = Image.fromarray(quantized_pixels)
    result_image.save(output_filename)
    print(f"Saved: {output_filename}")

def histogram(image_path, title):
    
    img = Image.open(image_path).convert("L")
    pixels = np.array(img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(pixels, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f"Image: {title}")
    axes[0].axis('off')

    axes[1].hist(pixels.ravel(), bins=256, range=(0, 256), color="black")
    axes[1].set_title(f"Histogram: {title}")
    axes[1].set_xlabel("Pixel Intensity (0 = Black, 255 = White)")
    axes[1].set_ylabel("Pixel Count")
    axes[1].set_xlim([0, 255])

    plt.tight_layout()
    plt.savefig(f"histogram_{title.replace('.jpg', '')}.png")
    plt.show()

def histogram_cdf(image_path):
    # Load image in 8-bit grayscale
    img = Image.open(image_path).convert("L")
    pixels = np.array(img)

    # Calculate the 256-bin histogram
    # hist contains pixel counts per bin; bin_edges has boundary values
    hist, bin_edges = np.histogram(pixels.ravel(), bins=256, range=(0, 256))

    # Calculate Normalized Histogram (Probability Mass Function)
    total_pixels = pixels.size
    pmf = hist / total_pixels
    
    # Calculate Cumulative Density Function (CDF)
    # np.cumsum calculates the running cumulative sum across the 256 bins
    cdf = np.cumsum(pmf)

    # Plot Image, Histogram, and CDF side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15 ,4))

    # The Image
    axes[0].imshow(pixels, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f"Image: {image_path}")
    axes[0].axis('off')

    # Histogram
    axes[1].bar(range(256), hist, color="black", width=1.0)
    axes[1].set_title(f"Histogram: {image_path}")
    axes[1].set_xlabel("Pixel Intensity (0 = Black, 255 = White)")
    axes[1].set_ylabel("Pixel Count")
    axes[1].set_xlim([0, 255])

    # Cumulative Density Function (CDF)
    axes[2].plot(range(256), cdf, color="blue", linewidth=2)
    axes[2].set_title(f"CDF: {image_path}")
    axes[2].set_xlabel("Pixel Intensity (0 = Black, 255 = White)")
    axes[2].set_ylabel("Cumulative Probability (0.0 to 1.0)")
    axes[2].set_xlim([0, 255])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"histogram_cdf_{image_path.replace('.jpg', '')}.png")
    plt.show()

def histogram_matching(source_path, reference_path):
    # 1. Load images as 8-bit grayscale arrays
    img_source = Image.open(source_path).convert("L")
    img_ref = Image.open(reference_path).convert("L")

    pixels_src = np.array(img_source)
    pixels_ref = np.array(img_ref)

    # 2. Compute histograms
    hist_src, _ = np.histogram(pixels_src.ravel(), bins=256, range=(0, 256))
    hist_ref, _ = np.histogram(pixels_ref.ravel(), bins=256, range=(0, 256))

    # 3. Compute Normalized CDFs
    cdf_src = np.cumsum(hist_src) / pixels_src.size
    cdf_ref = np.cumsum(hist_ref) / pixels_ref.size

    # 4. Construct the Mapping Lookup Table
    lookup_table = np.zeros(256, dtype=np.uint8)

    for src_val in range(256):
        # Find index in cdf_ref that is closest to cdf_src[src_val]
        diff = np.abs(cdf_ref - cdf_src[src_val])
        lookup_table[src_val] = np.argmin(diff)

    # 5. Apply the mapping to the source image pixels
    matched_pixels = lookup_table[pixels_src]
    matched_image = Image.fromarray(matched_pixels)
    matched_image.save("dark_flower_matched.png")

    # 6. Plot Original Source, Target Reference, and Matched Output with Histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Images (Top Row)
    axes[0, 0].imshow(pixels_src, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Source: {source_path}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pixels_ref, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Reference: {reference_path}")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(matched_pixels, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Result: Matched Output")
    axes[0, 2].axis("off")

    # Histograms (Bottom Row)
    axes[1, 0].hist(pixels_src.ravel(), bins=256, range=(0, 256), color="black")
    axes[1, 0].set_title("Source Histogram")

    axes[1, 1].hist(pixels_ref.ravel(), bins=256, range=(0, 256), color="black")
    axes[1, 1].set_title("Reference Histogram")

    hist_matched, _ = np.histogram(
        matched_pixels.ravel(), bins=256, range=(0, 256)
    )
    axes[1, 2].hist(
        matched_pixels.ravel(), bins=256, range=(0, 256), color="black"
    )
    axes[1, 2].set_title("Matched Histogram")

    plt.tight_layout()
    plt.savefig("histogram_matching_report.png")
    plt.show()

    return matched_image

while True:
    print("\nAvailable functions:")
    print("- size_and_color")
    print("- scaling_bilinear")
    print("- scaling")
    print("- grayscale")
    print("- color_quantization")
    print("- histogram")
    print("- histogram_cdf")
    print("- histogram_matching\n")

    user_input = input("Enter the assignment name (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the program.\n")
        break
    try:
        if user_input == size_and_color.__name__:
            size_and_color("linnanmaa.jpg")
        elif user_input == scaling.__name__:
            scaled_image = scaling("shirt_small.jpg", 4.0)
            scaled_image.show()
        elif user_input == scaling_bilinear.__name__:
            scaled_image = scaling_bilinear("shirt_small.jpg", 4.0)
            scaled_image.show()
        elif user_input == grayscale.__name__:
            grayscale("shirt.jpg")
        elif user_input == color_quantization.__name__:
            for bits in [4, 3, 2, 1]:
                color_quantization("linnanmaa.jpg", bits)
        elif user_input == histogram.__name__:
            histogram("flower.jpg", "flower.jpg")
            histogram("dark_flower.jpg", "dark_flower.jpg")            
        elif user_input == histogram_cdf.__name__:
            histogram_cdf("flower.jpg")
        elif user_input == histogram_matching.__name__:
            histogram_matching("dark_flower.jpg", "flower.jpg")
        else:
            print("Invalid input! Please enter one of the listed functions.")
    except FileNotFoundError:
        print(f"Function not found: {user_input}. Please check the name and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")