import qrcode
import numpy as np
import random
from PIL import Image
import cv2
import os
import sys
import math   ### ADDED

# Step X: Shannon Entropy function
def shannon_entropy(data):   ### ADDED
    """Calculate Shannon entropy for binary string/list/array."""
    if not isinstance(data, str):
        data = ''.join(str(x) for x in data)
    
    length = len(data)
    if length == 0:
        return 0.0

    freq = {}
    for symbol in data:
        freq[symbol] = freq.get(symbol, 0) + 1

    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy

# Memory size utility function
def get_memory_size(obj):
    return sys.getsizeof(obj)

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

# Step 3: Binary-to-Integer Compression
def binary_to_integer_compression(data, chunk_size=8):
    padding = (chunk_size - len(data) % chunk_size) % chunk_size
    data = data + '0' * padding
    compressed = [int(data[i:i + chunk_size], 2) for i in range(0, len(data), chunk_size)]
    return compressed, padding

# Integer-to-Binary Decompression
def integer_to_binary_decompression(compressed_data, padding, chunk_size=8):
    binary_str = ''.join([format(num, '0{}b'.format(chunk_size)) for num in compressed_data])
    # Remove padding
    if padding > 0:
        binary_str = binary_str[:-padding]
    return binary_str

# Save integer data in text format
def save_integer_to_text(data, filename):
    with open(filename, 'w', encoding="utf-8") as file:
        file.write(' '.join(map(str, data)))  # Store integer values as space-separated numbers
    print(f"Integer data saved to {filename}. Number of characters: {os.path.getsize(filename)}")

# Read integer data from text format
def read_integer_from_text(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        data = file.read().split()
    return list(map(int, data))

# Embed compressed data into a QR Code (Commented as requested)
'''
def embed_compressed_data_in_qr(data, filename="compressed_qr.png"):
    qr_data = ' '.join(map(str, data))  # Convert integers to a space-separated string
    qr_array = generate_qr(qr_data, filename)
    print(f"Compressed data embedded in QR Code: {filename}")
    return qr_array
'''

# Main Workflow
if __name__ == "__main__":
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")

     ### ADDED: Original QR entropy analysis
    qr_binary = "".join("1" if bit else "0" for row in qr_array for bit in row)
    qr_zeros = qr_binary.count("0")
    qr_ones = qr_binary.count("1")
    qr_entropy = shannon_entropy(qr_binary)
    print("\n=== Original QR Code Binary ===")
    print(f"Size: {len(qr_binary)} bits")
    print(f"Number of 0s: {qr_zeros}, Number of 1s: {qr_ones}")
    print(f"Shannon Entropy: {qr_entropy:.4f} bits/symbol\n")

    share1, share2 = create_shares(qr_array)
    save_share_as_image(share1, "share1.png")
    save_share_as_image(share2, "share2.png")


    binary_data = ''.join(map(str, share1.ravel()))
    print(f"Size of binary data before compression: {len(binary_data)} bits")

    ### ADDED: Count zeros and ones
    zeros = binary_data.count("0")
    ones = binary_data.count("1")
    print(f"Number of 0s: {zeros}, Number of 1s: {ones}")

    ### ADDED: Shannon Entropy
    entropy_value = shannon_entropy(binary_data)
    print(f"Shannon Entropy (before compression): {entropy_value:.4f} bits/symbol")

    if abs(entropy_value - 1.0) < 0.05:
        print("Data is HIGH entropy (random-like, less compressible).")
    elif entropy_value > 0.7:
        print("Data has MODERATE entropy (some patterns, partially compressible).")
    else:
        print("Data has LOW entropy (predictable, highly compressible).")
    
    compressed_data, padding = binary_to_integer_compression(binary_data)
    save_integer_to_text(compressed_data, "compressed_share1.txt")
    
    print(f"Number of characters in compressed text file: {os.path.getsize('compressed_share1.txt')}")
    
    # QR embedding commented out as requested
    # embed_compressed_data_in_qr(compressed_data, "compressed_qr.png")
    
    # Inverse process to recover original bits
    # Read compressed data from text file
    compressed_data_read = read_integer_from_text("compressed_share1.txt")
    
    # Decompress to binary
    recovered_binary = integer_to_binary_decompression(compressed_data_read, padding)
    
    print(f"Size of recovered binary data: {len(recovered_binary)} bits")
    
    # Verify if the recovered binary matches the original
    if recovered_binary == binary_data:
        print("Success: Recovered binary data matches original!")
    else:
        print("Error: Recovered binary data does NOT match original!")