import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct


# ==========================
# LOAD IMAGE
# ==========================

img = Image.open("photo.bmp").convert("L")
image = np.array(img)

print("Image shape:", image.shape)


# ==========================
# SPLIT INTO 8x8 BLOCKS
# ==========================

blocks = []

for y in range(0, 640, 8):
    for x in range(0, 640, 8):
        blocks.append(image[y:y+8, x:x+8])

print("Number of blocks:", len(blocks))


# ==========================
# DCT FUNCTIONS
# ==========================

def dct2(block):
    return dct(
        dct(block.T, norm="ortho").T,
        norm="ortho"
    )


def idct2(block):
    return idct(
        idct(block.T, norm="ortho").T,
        norm="ortho"
    )


# ==========================
# ZIGZAG
# ==========================

def zigzag_indices(n=8):

    indices = []

    for s in range(2 * n - 1):

        if s % 2 == 0:

            for i in range(s + 1):

                row = i
                col = s - i

                if row < n and col < n:
                    indices.append((row, col))

        else:

            for i in range(s + 1):

                row = s - i
                col = i

                if row < n and col < n:
                    indices.append((row, col))

    return indices


ZZ = zigzag_indices()


def zigzag_scan(block):

    return np.array(
        [block[r, c] for r, c in ZZ]
    )


def inverse_zigzag(vector):

    block = np.zeros((8, 8), dtype=vector.dtype)

    for value, (r, c) in zip(vector, ZZ):
        block[r, c] = value

    return block


# ZigZag test

test_block = np.arange(64).reshape(8, 8)

z = zigzag_scan(test_block)

restored = inverse_zigzag(z)

print(
    "ZigZag test:",
    np.array_equal(test_block, restored)
)


# ==========================
# RLE
# ==========================

def rle_encode(data):

    encoded = []

    i = 0

    while i < len(data):

        if data[i] == 0:

            count = 1

            while (
                i + count < len(data)
                and data[i + count] == 0
            ):
                count += 1

            encoded.append(0)
            encoded.append(count)

            i += count

        else:

            encoded.append(data[i])

            i += 1

    return np.array(encoded, dtype=np.int8)


def rle_decode(encoded, expected_length=64):

    decoded = []

    i = 0

    while len(decoded) < expected_length:

        value = encoded[i]

        if value == 0:

            count = encoded[i + 1]

            decoded.extend([0] * count)

            i += 2

        else:

            decoded.append(value)

            i += 1

    return np.array(decoded, dtype=np.int8)


# RLE test

test = np.array([
    1, 5, 0, 6, 0, 0, 0, 0, 2
])

encoded = rle_encode(test)

decoded = rle_decode(
    encoded,
    expected_length=len(test)
)

print(
    "RLE test:",
    np.array_equal(test, decoded)
)


# ==========================
# QUANTIZATION VALUE
# ==========================

Q_values = [16, 32, 64, 128]

for Q in Q_values:

    print("\n" + "=" * 50)
    print(f"Processing Q = {Q}")
    print("=" * 50)

    # ==========================
    # COMPRESSION
    # ==========================

    quantized_blocks = []

    for block in blocks:

        shifted = block.astype(np.float32) - 128

        dct_block = dct2(shifted)

        quantized = np.round(dct_block / Q)

        quantized = np.clip(
            quantized,
            -128,
            127
        ).astype(np.int8)

        quantized_blocks.append(quantized)

    all_compressed = []

    for block in quantized_blocks:

        zz = zigzag_scan(block)

        rle = rle_encode(zz)

        all_compressed.extend(rle)

    compressed = np.array(
        all_compressed,
        dtype=np.int8
    )

    # Save compressed file

    compressed_filename = f"compressed_Q{Q}.bin"

    compressed.tofile(compressed_filename)

    compressed_from_file = np.fromfile(
        compressed_filename,
        dtype=np.int8
    )

    # ==========================
    # COMPRESSION RATIO
    # ==========================

    original_size = image.nbytes
    compressed_size = compressed.nbytes

    ratio = original_size / compressed_size

    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {ratio:.2f}:1")

    # ==========================
    # DECOMPRESSION
    # ==========================

    reconstructed = np.zeros(
        (640, 640),
        dtype=np.uint8
    )

    position = 0

    for y in range(0, 640, 8):

        for x in range(0, 640, 8):

            decoded = []

            while len(decoded) < 64:

                value = compressed_from_file[position]

                if value == 0:

                    count = compressed_from_file[position + 1]

                    decoded.extend([0] * count)

                    position += 2

                else:

                    decoded.append(value)

                    position += 1

            decoded = np.array(
                decoded[:64],
                dtype=np.int16
            )

            quantized_block = inverse_zigzag(decoded).astype(np.int16)

            dct_block = quantized_block * Q

            reconstructed_block = idct2(dct_block)

            reconstructed_block += 128

            reconstructed_block = np.clip(
                np.round(reconstructed_block),
                0,
                255
            ).astype(np.uint8)

            reconstructed[
                y:y+8,
                x:x+8
            ] = reconstructed_block

    # Save reconstructed image

    output_filename = f"reconstructed_Q{Q}.bmp"

    Image.fromarray(
        reconstructed
    ).save(output_filename)

    Image.fromarray(
        reconstructed
    ).save("reconstructed.bmp")

    print(f"Saved: {output_filename}")