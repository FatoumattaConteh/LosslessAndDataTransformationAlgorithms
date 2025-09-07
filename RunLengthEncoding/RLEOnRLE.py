import qrcode
import numpy as np
import random
from PIL import Image
import json
import os
import struct
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

# Step 1: Generate a QR Code
def generate_qr(data, filename="qr_code.png"):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)

# Step 2: Create Visual Cryptography Shares
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

# Save share as an image
def save_share_as_image(share, filename):
    img = Image.fromarray((share * 255).astype(np.uint8))
    img.save(filename, format="PNG")

# Run-Length Encoding (RLE)
def run_length_encoding(data):
    encoded = []
    if len(data) == 0:
        return encoded
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append((int(data[i - 1]), count))
            count = 1
    encoded.append((int(data[-1]), count))
    return encoded

# Run-Length Decoding (RLE)
def run_length_decoding(encoded):
    decoded = []
    for value, count in encoded:
        decoded.extend([value] * count)
    return decoded

# Save encoded vector to binary file (pairs -> bytes)
def save_vector_to_binary(vector, filename):
    with open(filename, 'wb') as file:
        for value, count in vector:
            file.write(struct.pack('BB', value, min(count, 255)))

# Save encoded vector to flat text file: "v1 c1 v2 c2 ..."
def save_vector_to_text(vector, filename):
    flat = []
    for value, count in vector:
        flat.append(str(value))
        flat.append(str(count))
    encoded_string = ' '.join(flat)
    with open(filename, 'w') as file:
        file.write(encoded_string)
    return encoded_string

# Load flat text and convert to pairs
def load_rle_from_text(filename):
    with open(filename, 'r') as file:
        s = file.read().strip()
        if not s:
            return []
        numbers = list(map(int, s.split()))
    if len(numbers) % 2 != 0:
        raise ValueError("Loaded RLE text has odd number of integers; expected value,count pairs.")
    return [(numbers[i], numbers[i+1]) for i in range(0, len(numbers), 2)]

# Reconstruct QR from shares
def reconstruct_qr(share1, share2):
    return np.bitwise_xor(share1, share2)

# Utility: flatten pairs into flat list [v,c,v,c,...]
def pairs_to_flat(pairs):
    flat = []
    for v,c in pairs:
        flat.append(v)
        flat.append(c)
    return flat

# Utility: convert flat list [v,c,...] into pairs
def flat_to_pairs(flat):
    if len(flat) % 2 != 0:
        raise ValueError("Flat list cannot be converted into pairs (odd length).")
    return [(int(flat[i]), int(flat[i+1])) for i in range(0, len(flat), 2)]

# Main workflow
if __name__ == "__main__":
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")

    # --- Entropy after QR ---
    qr_flat = qr_array.ravel().tolist()
    qr_entropy = shannon_entropy(qr_flat)
    print(f"[QR] Size: {len(qr_flat)} bits | Entropy: {qr_entropy:.4f}")

    # Create and save shares
    share1, share2 = create_shares(qr_array)
    save_share_as_image(share1, "share1.png")
    save_share_as_image(share2, "share2.png")

    # --- Entropy after VC ---
    share1_flat = share1.ravel().tolist()
    vc_entropy = shannon_entropy(share1_flat)
    print(f"[VC Share1] Size: {len(share1_flat)} bits | Entropy: {vc_entropy:.4f}")

    # Prepare for iterative compression
    vector1 = share1_flat[:]  # list
    encoded_vectors = []

    print("\n--- Iterative RLE Compression (10 rounds) ---\n")
    for i in range(10):
        encoded_vector = run_length_encoding(vector1)   # list of pairs
        encoded_vectors.append(encoded_vector)

        # Save iteration results
        binary_filename = f"rle_{i+1}_encoded_vector.bin"
        text_filename = f"rle_{i+1}_encoded_vector.txt"

        save_vector_to_binary(encoded_vector, binary_filename)
        save_vector_to_text(encoded_vector, text_filename)

        # Stats
        binary_size_bits = os.path.getsize(binary_filename) * 8
        loaded_pairs = load_rle_from_text(text_filename)
        text_size_elements = len(loaded_pairs)
        char_count = len(''.join(ch for ch in json.dumps(loaded_pairs) if ch not in " ,:[]"))

        print(f"Iteration {i+1}:")
        print(f"  Binary file: {binary_filename}, size = {binary_size_bits} bits")
        print(f"  Text file: {text_filename}, pairs = {text_size_elements}")
        print(f"  Character count (filtered) = {char_count}\n")

        # flatten for next round (v,c,v,c,...)
        vector1 = pairs_to_flat(encoded_vector)

    # --- Iterative RLE Decompression (reverse) ---
    print("\n--- Iterative RLE Decompression (reverse) ---\n")

    # start from the 10th encoded pairs
    current = encoded_vectors[-1]   # list of pairs
    final_flat = None

    for stage in range(10, 0, -1):
        # ensure current is pairs
        if all(isinstance(el, (list, tuple)) and len(el) == 2 for el in current):
            pairs = [(int(a), int(b)) for a, b in current]
        else:
            # maybe current is flat ints
            if all(isinstance(x, (int, np.integer)) for x in current):
                pairs = flat_to_pairs(current)
            else:
                # try to coerce strings to ints (if loaded from file etc.)
                try:
                    ints = [int(x) for x in current]
                    pairs = flat_to_pairs(ints)
                except Exception as e:
                    raise TypeError(f"Cannot interpret current encoded structure at stage {stage}: {e}")

        # decode pairs -> flat bits
        decoded_flat = run_length_decoding(pairs)
        print(f"After decoding iteration {stage}: length = {len(decoded_flat)}")

        if stage == 1:
            final_flat = decoded_flat
            break

        # Otherwise re-wrap decoded_flat into pairs for next reverse step
        if len(decoded_flat) % 2 != 0:
            # This can happen if the forward compression flattened into an odd-length sequence;
            # that is unexpected for the pipeline you described.
            raise ValueError(f"Decoded flat at stage {stage} has odd length ({len(decoded_flat)}); cannot re-pair.")
        current = flat_to_pairs(decoded_flat)

    # final_flat must be the full decompressed 0/1 sequence
    if final_flat is None:
        raise RuntimeError("Decompression did not produce final flattened vector.")

    # Save the 10th decompressed flat vector to text for inspection
    with open("decompressed_from_iter10.txt", "w") as f:
        f.write(" ".join(map(str, final_flat)))

    # Reshape to share shape and compare
    decompressed_share = np.array(final_flat, dtype=np.uint8).reshape(share1.shape)

    # --- Entropy after decompression ---
    decompressed_entropy = shannon_entropy(final_flat)
    print(f"\n[Decompressed Share1] Size: {len(final_flat)} bits | Entropy: {decompressed_entropy:.4f}")

    # Compare decompressed share with original share1
    if np.array_equal(share1, decompressed_share):
        print("✅ SUCCESS: 10th decompressed share matches original share1 exactly.")
    else:
        diff = np.sum(share1 != decompressed_share)
        print(f"❌ ERROR: 10th decompressed share does NOT match original. Differences: {diff}")

    # Verify QR reconstruction
    reconstructed_qr = reconstruct_qr(decompressed_share, share2)
    Image.fromarray((reconstructed_qr * 255).astype(np.uint8)).save("reconstructed_qr_from_iter10.png")

    if np.array_equal(qr_array, reconstructed_qr):
        print("✅ QR reconstruction matches original QR.")
    else:
        print("❌ QR reconstruction failed.")
