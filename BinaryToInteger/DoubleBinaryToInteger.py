import qrcode
import numpy as np
import random
import json
import os
from PIL import Image
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

def generate_qr(data, filename="qr_code.png"):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)

def create_shares(qr_array):
    rows, cols = qr_array.shape
    share1 = np.zeros((rows, cols), dtype=np.uint8)
    share2 = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            pattern = random.choice([[0, 0], [1, 1]]) if qr_array[i, j] == 0 else random.choice([[0, 1], [1, 0]])
            share1[i, j], share2[i, j] = pattern
    return share1, share2

def binary_to_integer_compression(binary_data):
    chunk_size = 8
    padding = (chunk_size - (len(binary_data) % chunk_size)) % chunk_size
    binary_data += "0" * padding
    compressed_data = [int(binary_data[i:i + chunk_size], 2) for i in range(0, len(binary_data), chunk_size)]
    return compressed_data, padding

def integer_to_binary_decompression(compressed_data, padding):
    binary_data = ''.join(f"{int(x):08b}" for x in compressed_data)
    return binary_data[:-padding] if padding else binary_data

def save_data_to_file(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)
    print(f"File '{filename}' size: {os.path.getsize(filename)} characters")

def save_binary_data_to_file(data, filename):
    with open(filename, "wb") as file:
        file.write(bytearray(data))
    print(f"Binary File '{filename}' size: {os.path.getsize(filename) * 8} bits")

def reconstruct_qr(share1, share2):
    return np.where(share1 == share2, 0, 1).astype(np.uint8)

def read_compressed_data(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data["data"], data["padding"]

# Main execution
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

binary_data = ''.join(map(str, share1.ravel()))
print(f"Initial binary data size: {len(binary_data)} bits")

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
    

compressed_data = binary_data
paddings = []

# Compression process
for i in range(10):
    if isinstance(compressed_data, list):  
        compressed_data = ''.join(f"{x:08b}" for x in compressed_data)  

    compressed_data, padding = binary_to_integer_compression(compressed_data)
    paddings.append(padding)

    txt_filename = f"compressed_share1_level_{i+1}.txt"
    json_filename = f"compressed_share1_level_{i+1}.json"
    
    with open(txt_filename, "w") as file:
        file.write(" ".join(map(str, compressed_data)))
    print(f"File '{txt_filename}' size with spaces: {os.path.getsize(txt_filename)} characters")
    print(f"Number of characters in '{txt_filename}': {len(compressed_data)} characters")

    save_data_to_file({"data": compressed_data, "padding": padding}, json_filename)
    binary_filename = f"compressed_share1_level_{i+1}.bin"
    save_binary_data_to_file(compressed_data, binary_filename)

# Comment out QR embedding as requested
'''
final_json = json.dumps({"data": compressed_data, "paddings": paddings})
generate_qr(final_json, "compressed_qr.png")
'''

# Decompression process (independent of QR embedding)
# Load the final compressed data and paddings
final_json_filename = "compressed_share1_level_10.json"
compressed_data, padding = read_compressed_data(final_json_filename)

# We need all paddings for decompression, so we'll load them from all files
all_paddings = []
for i in range(10):
    json_filename = f"compressed_share1_level_{i+1}.json"
    with open(json_filename, "r") as file:
        data = json.load(file)
    all_paddings.append(data["padding"])

# Decompress through all levels
decompressed_data = compressed_data
for i in range(9, -1, -1):
    decompressed_binary = integer_to_binary_decompression(decompressed_data, all_paddings[i])
    if i > 0:
        # Convert back to integer list for next level decompression
        decompressed_data = [int(decompressed_binary[j:j+8], 2) for j in range(0, len(decompressed_binary), 8)]
    else:
        # Final level - convert to binary array
        share1_restored = np.array(list(map(int, decompressed_binary)), dtype=np.uint8).reshape(share1.shape)

# Verify the decompressed data matches the original
original_binary = ''.join(map(str, share1.ravel()))
decompressed_binary = ''.join(map(str, share1_restored.ravel()))

if original_binary == decompressed_binary:
    print("SUCCESS: Decompressed data matches original exactly!")
    print(f"Original size: {len(original_binary)} bits")
    print(f"Decompressed size: {len(decompressed_binary)} bits")
else:
    print("ERROR: Decompressed data does not match original!")
    # Count differences
    differences = sum(1 for a, b in zip(original_binary, decompressed_binary) if a != b)
    print(f"Number of differing bits: {differences}")

# Reconstruct QR code from shares
reconstructed_qr = reconstruct_qr(share1_restored, share2)
Image.fromarray((reconstructed_qr * 255).astype(np.uint8)).save("reconstructed_qr.png")
print("Reconstruction complete. Check 'reconstructed_qr.png'.")