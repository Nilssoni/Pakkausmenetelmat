from PIL import Image
import numpy as np
import matplotlib

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
    scaled_image.save(f"scaled_{image_path}")
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

while True:
    print("\nAvailable functions:")
    print("- size_and_color")
    print("- scaling")
    print("- grayscale")
    print("- color_quantization\n")
    user_input = input("Enter the assignment name (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the program.\n")
        break
    try:
        if user_input == size_and_color.__name__:
            size_and_color("linnanmaa.jpg")
        elif user_input == scaling.__name__:
            scaled_image = scaling("shirt.jpg", 0.17)
            scaled_image.show()
        elif user_input == grayscale.__name__:
            grayscale("shirt.jpg")
        elif user_input == color_quantization.__name__:
            for bits in [4, 3, 2, 1]:
                color_quantization("linnanmaa.jpg", bits)
        else:
            print("Invalid input! Please enter one of the listed functions.")
    except FileNotFoundError:
        print(f"Function not found: {user_input}. Please check the name and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")