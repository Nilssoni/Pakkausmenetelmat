import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ass_4b():
    fs = 100_000 # 100 Khz frequency
    duration = 0.01 # 10 ms

    N = int(fs * duration) # number of samples
    t = np.arange(N) / fs # time vector

    # 1 Khz sine wave
    sine_1K = np.sin(2 * np.pi * 1000 * t)

    # 1 Khz square wave
    square_1k = np.sign(np.sin(2 * np.pi * 1000 * t))

    # 1 Khz + 8 Khz sine wave
    sum_signal = (
        np.sin(2 * np.pi * 1000 * t)
        + np.sin(2 * np.pi * 8000 * t)
    )

    # FFT
    fft_sine = np.fft.fft(sine_1K)
    fft_square = np.fft.fft(square_1k)
    fft_sum = np.fft.fft(sum_signal)

    # Frequency 
    freqs = np.fft.fftfreq(N, 1/fs)

    # Magnitude spectrum
    magnitude = np.abs(fft_sine) / N

    magnitude_square = np.abs(fft_square) / N
    magnitude_sum = np.abs(fft_sum) / N

    half = N // 2

    freqs = freqs[:half]
    magnitude = magnitude[:half]
    magnitude_square = magnitude_square[:half]
    magnitude_sum = magnitude_sum[:half]

    # Plot
    fig, ax = plt.subplots(3, 2, figsize=(14, 10))

    # 1 kHz sine
    ax[0, 0].plot(t, sine_1K)
    ax[0, 0].set_title("1 kHz Sine Wave")

    ax[0, 1].plot(freqs, magnitude)
    ax[0, 1].set_title("FFT")
    ax[0, 1].set_xlim(0, 15000)

    # 1 kHz square
    ax[1, 0].plot(t, square_1k)
    ax[1, 0].set_title("1 kHz Square Wave")

    ax[1, 1].plot(freqs, magnitude_square)
    ax[1, 1].set_title("FFT")
    ax[1, 1].set_xlim(0, 15000)

    # 1 kHz + 8 kHz
    ax[2, 0].plot(t, sum_signal)
    ax[2, 0].set_title("1 kHz + 8 kHz Signal")

    ax[2, 1].plot(freqs, magnitude_sum)
    ax[2, 1].set_title("FFT")
    ax[2, 1].set_xlim(0, 15000)

    plt.tight_layout()
    plt.show()

def ass_4c():

    def show_fft(filename):
        img = Image.open(filename)
        gray = np.array(img.convert("L"))

        F = np.fft.fft2(gray)
        F_shifted = np.fft.fftshift(F)

        spectrum = np.log(1 + np.abs(F_shifted))

        return gray, spectrum

    vl_img, vl_spec = show_fft("vl.png")
    hl_img, hl_spec = show_fft("hl.png")
    diag_img, diag_spec = show_fft("diag.png")

    fig, ax = plt.subplots(3, 2, figsize=(10, 12))

    # Vertical lines
    ax[0, 0].imshow(vl_img, cmap="gray")
    ax[0, 0].set_title("vl.png")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(vl_spec, cmap="gray")
    ax[0, 1].set_title("FFT Spectrum")
    ax[0, 1].axis("off")

    # Horizontal lines
    ax[1, 0].imshow(hl_img, cmap="gray")
    ax[1, 0].set_title("hl.png")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(hl_spec, cmap="gray")
    ax[1, 1].set_title("FFT Spectrum")
    ax[1, 1].axis("off")

    # Diagonal lines
    ax[2, 0].imshow(diag_img, cmap="gray")
    ax[2, 0].set_title("diag.png")
    ax[2, 0].axis("off")

    ax[2, 1].imshow(diag_spec, cmap="gray")
    ax[2, 1].set_title("FFT Spectrum")
    ax[2, 1].axis("off")

    plt.tight_layout()
    plt.show()

def ass_4d():
    img = Image.open("fox.jpg")
    gray = np.array(img.convert("L"))

    F = np.fft.fft2(gray)
    F_shifted = np.fft.fftshift(F)

    rows, cols = gray.shape

    crow = rows // 2
    ccol = cols // 2

    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow)**2 + (X - ccol)**2)

    mask = distance <= 50
    F_filtered = F_shifted * mask
    spectrum_original = np.log(1 + np.abs(F_shifted))
    spectrum_filtered = np.log(1 + np.abs(F_filtered))
    F_inverse_shift = np.fft.ifftshift(F_filtered)
    filtered_img = np.fft.ifft2(F_inverse_shift)
    filtered_img = np.abs(filtered_img)

    fig, ax = plt.subplots(2, 2, figsize=(12,10))

    # Original image
    ax[0,0].imshow(gray, cmap="gray")
    ax[0,0].set_title("Original Image")
    # Original spectrum
    ax[0,1].imshow(spectrum_original, cmap="gray")
    ax[0,1].set_title("Original Spectrum")
    # Filtered image
    ax[1,0].imshow(filtered_img, cmap="gray")
    ax[1,0].set_title("Filtered Image")
    # Filtered spectrum
    ax[1,1].imshow(spectrum_filtered, cmap="gray")
    ax[1,1].set_title("Filtered Spectrum")

    for row in ax:
        for a in row:
            a.axis("off")

    plt.tight_layout()
    plt.show()



while True:
    try:
        print("\nAvailable assignments:")
        print("1: Assignment 4b")
        print("2: Assignment 4c")
        print("3: Assignment 4d")
        choice = str(input("Enter the assignment number (or type 'exit' to quit): "))
        if choice.lower() == 'exit':
            print("Exiting the program.\n")
            break
        elif choice == '1':
            ass_4b()
            break
        elif choice == '2':
            ass_4c()
            break
        elif choice == '3':
            ass_4d()
            break
        else:
            print("Invalid choice. Please enter '1' or '2'.")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"An error occurred: {e}")
    