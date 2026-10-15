import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
# d:/codes/Pakkausmenetelmät/.venv/Scripts/Activate.ps1


def assignment_5a():
    N = 8
    n = np.arange(8)

    def dct_basis(k, N):
        if k == 0:
            alpha = np.sqrt(1 / N)
        else:
            alpha = np.sqrt(2 / N)
        return alpha * np.cos(
            (np.pi / N) * (n + 0.5) * k
        )

    def dft_basis(k, N):
        return np.exp(-1j * 2 * np.pi * k * n / N)
    #  DCT basis function
    dct_k0 = dct_basis(0, N)
    dct_k1 = dct_basis(1, N)
    #  DFT basis function
    dft_k0 = dft_basis(0, N)
    dft_k1 = dft_basis(1, N)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    # DCT
    axes[0, 0].stem(n, dct_k0)
    axes[0, 0].set_title("DCT Basis k=0")
    axes[0, 0].set_xlabel("n")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)

    axes[0, 1].stem(n, dct_k1)
    axes[0, 1].set_title("DCT Basis k=1")
    axes[0, 1].set_xlabel("n")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True)
    # DFT
    axes[1, 0].stem(n, np.real(dft_k0))
    axes[1, 0].set_title("DFT Real Part k=0")
    axes[1, 0].set_xlabel("n")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True)

    axes[1, 1].stem(n, np.real(dft_k1))
    axes[1, 1].set_title("DFT Real Part k=1")
    axes[1, 1].set_xlabel("n")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True)

    # DFT Imag + Magnitude
    axes[2, 0].stem(n, np.imag(dft_k1))
    axes[2, 0].set_title("DFT Imaginary Part k=1")
    axes[2, 0].set_xlabel("n")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].grid(True)

    axes[2, 1].stem(n, np.abs(dft_k1))
    axes[2, 1].set_title("DFT Magnitude k=1")
    axes[2, 1].set_xlabel("n")
    axes[2, 1].set_ylabel("Magnitude")
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.show()

def assignment_5b():
    x = np.array([0,1,2,3,4,5,6,7])

    dct_result = dct(x, norm='ortho')
    dft_result = np.fft.fft(x)

    fig, ax = plt.subplots(3, 1, figsize=(8,8))

    ax[0].stem(dct_result)
    ax[0].set_title("DCT")

    ax[1].stem(np.real(dft_result))
    ax[1].set_title("DFT Real Part")

    ax[2].stem(np.imag(dft_result))
    ax[2].set_title("DFT Imaginary Part")

    plt.tight_layout()
    plt.show()

def assignment_5c():
    # Original signal
    x = np.arange(8)

    # DCT
    dct_result = dct(x, norm='ortho')
    dct_filtered = dct_result.copy()

    # Remove half of coefficients with smallest magnitude
    dct_indices = np.argsort(np.abs(dct_filtered))
    dct_filtered[dct_indices[:4]] = 0

    # Reconstruct signal
    x_dct = idct(dct_filtered, norm='ortho')

    # DFT
    dft_result = np.fft.fft(x)
    dft_filtered = dft_result.copy()

    # Keep only 4 largest magnitude coefficients
    keep = np.argsort(np.abs(dft_result))[-4:]
    mask = np.zeros(len(dft_result), dtype=bool)
    mask[keep] = True
    dft_filtered[~mask] = 0

    # Reconstruct signal
    x_dft = np.real(np.fft.ifft(dft_filtered))
    plt.figure(figsize=(8, 5))

    plt.plot(x, 'ko-', label='Original')
    plt.plot(x_dct, 'bs--', label='DCT Reconstructed') # type: ignore
    plt.plot(x_dft, 'r^--', label='DFT Reconstructed')

    plt.title("Effect of Zeroing Half of Frequency Components")
    plt.xlabel("Sample")
    plt.ylabel("Value")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def assignment_5d():

    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.fftpack import dct

    def dct2(image):
        return dct(
            dct(image, axis=0, norm='ortho'),
            axis=1,
            norm='ortho'
        )

    filenames = [
        "smooth.png",
        "vertical_stripes.png",
        "checkerboard.png"
    ]

    titles = [
        "Smooth",
        "Vertical Stripes",
        "Checkerboard"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, filename in enumerate(filenames):

        img = Image.open(filename)
        gray = np.array(img.convert("L"))

        dct_image = dct2(gray)

        # Log scaling for visibility
        spectrum = np.log1p(np.abs(dct_image))

        # Original image
        axes[0, i].imshow(gray, cmap="gray")
        axes[0, i].set_title(f"{titles[i]} Original")
        axes[0, i].axis("off")

        # DCT spectrum
        axes[1, i].imshow(spectrum, cmap="gray")
        axes[1, i].set_title(f"{titles[i]} DCT Spectrum")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

while True:
    try:
        print("\nAvailable assignments:")
        print("1: Assignment 5a")
        print("2: Assignment 5b")
        print("3: Assignment 5c")
        print("4: Assignment 5d")
        choice = str(input("Enter the assignment number (or type 'exit' to quit): "))
        if choice.lower() == 'exit':
            print("Exiting the program.\n")
            break
        elif choice == '1':
            assignment_5a()
            break
        elif choice == '2':
            assignment_5b()
            break
        elif choice == '3':
            assignment_5c()
            break
        elif choice == '4':
            assignment_5d()
            break
        else:
            print("Invalid choice. Please enter '1' or '2', etc...")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"An error occurred: {e}")