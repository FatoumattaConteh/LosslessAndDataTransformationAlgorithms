import qrcode
import random
import numpy as np
from PIL import Image
import math

# --- Shannon Entropy ---
def shannon_entropy(data):
    """Calculate Shannon entropy for binary data."""
    if isinstance(data, np.ndarray):
        data = data.ravel().tolist()
    if not data:
        return 0.0
    freq = {}
    for symbol in data:
        freq[symbol] = freq.get(symbol, 0) + 1
    entropy = 0.0
    length = len(data)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

# Generate QR Code
def generate_qr_code(data, filename="qr_code.png"):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)

# Create visual cryptography shares
def create_shares(qr_array):
    rows, cols = qr_array.shape
    share1 = np.zeros((rows, cols), dtype=np.uint8)
    share2 = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if qr_array[i, j] == 0:  # Black pixel
                pattern = random.choice([[0, 0], [1, 1]])
            else:  # White pixel
                pattern = random.choice([[0, 1], [1, 0]])
            share1[i, j] = pattern[0]
            share2[i, j] = pattern[1]

    return share1, share2


# LZW Compression
def lzw_compress(uncompressed):
    dictionary = {chr(i): i for i in range(256)}
    current_sequence = ""
    compressed_data = []
    dict_size = 256

    for symbol in uncompressed:
        sequence = current_sequence + symbol
        if sequence in dictionary:
            current_sequence = sequence
        else:
            compressed_data.append(dictionary[current_sequence])
            dictionary[sequence] = dict_size
            dict_size += 1
            current_sequence = symbol

    if current_sequence:
        compressed_data.append(dictionary[current_sequence])
    return compressed_data

# LZW Decompression
def lzw_decompress(compressed):
    dictionary = {i: chr(i) for i in range(256)}
    dict_size = 256
    result = []
    prev_code = compressed.pop(0)
    result.append(dictionary[prev_code])
    current_sequence = dictionary[prev_code]

    for code in compressed:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = current_sequence + current_sequence[0]
        else:
            raise ValueError("Bad compressed code: %s" % code)

        result.append(entry)
        dictionary[dict_size] = current_sequence + entry[0]
        dict_size += 1
        current_sequence = entry

    return ''.join(result)

# Save helper functions
def save_image(array, filename):
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(filename, format="PNG")


def save_text(filename, data):
    with open(filename, 'w') as f:
        f.write(data)
    print(f"{filename} saved with {len(data)} characters.")

# Save the LZW Compressed Data as Text (not binary)
def save_list(filename, data):
    with open(filename, 'w') as f:
        f.write(' '.join(map(str, data)))  # Join data with spaces (as text)
    print(f"{filename} saved with {len(data)} entries.")

# Main pipeline
def qr_vc_pipeline(data):
    print("\n--- Starting QR-VC Pipeline ---")

    # Step 1: Generate and save QR code
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr_code(data, "original_qr.png")

    # --- Entropy analysis ---
    qr_binary = "".join("1" if bit else "0" for row in qr_array for bit in row)
    qr_entropy = shannon_entropy(qr_binary)
    print("\n=== Original QR Code Binary ===")
    print(f"Size: {len(qr_binary)} bits")
    print(f"Number of 0s: {qr_binary.count('0')}, Number of 1s: {qr_binary.count('1')}")
    print(f"Shannon Entropy: {qr_entropy:.4f} bits/symbol\n")

    # Step 2: Create shares
    share1, share2 = create_shares(qr_array)
    save_image(share1, "share1.png")
    save_image(share2, "share2.png")
    size= share1.shape
    print(f"Size of share1: {size}")
    
    # Share stats
    vector1 = [int(x) for x in share1.ravel()]
    print(f"Original share size: {len(vector1)} elements")
    entropy_share1 = shannon_entropy(share1)
    print("\n--- Share1 Statistics ---")
    print(f"Share1 size: {share1.size} bits")
    print(f"Number of 0s: {np.count_nonzero(share1 == 0)}, Number of 1s: {np.count_nonzero(share1 == 1)}")
    print(f"Shannon Entropy (Share1): {entropy_share1:.4f} bits/symbol")

    # Step 3: Flatten share1 and save as text
    flat_share1 = share1.flatten()
    flat_text = ''.join(map(str, flat_share1))
    save_text("share1_flattened.txt", flat_text)
    print(f"Characters in original Share 1: {len(flat_text)}")

    # Step 4: LZW Compression Round 1
    compressed1 = lzw_compress(flat_text)
    save_list("share1_lzw1.txt", compressed1)
    print(f"Integer chunks in LZW Compressed Share 1 (Round 1): {len(compressed1)}")
    compressed_str = ' '.join(map(str, compressed1))
    print(f"Characters in LZW Compressed Share 1 (Round 1): {len(compressed_str)}")


    # Step 5: Decompress back
    decompressed = lzw_decompress(compressed1.copy())  # Avoid modifying the original
    save_text("share1_decompressed.txt", decompressed)
    print(f"Characters in Decompressed Share 1: {len(decompressed)}")

    # Step 6: Verify reconstruction
    recovered = np.array(list(map(int, decompressed)), dtype=np.uint8).reshape(share1.shape)
    print(f"Decompressed Share 1 matches original: {np.array_equal(share1, recovered)}")

# Run it
qr_vc_pipeline("Hello from QR-VC!")
