import qrcode
import random
import numpy as np
from PIL import Image
import json
import os
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

# Read list from file
def read_list(filename):
    with open(filename, 'r') as f:
        data = f.read().split()
    return list(map(int, data))

# Save JSON data
def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Load JSON data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# QR Code embedding and extraction functions
def embed_in_qr(data, filename="compressed_qr.png"):
    """Embed data in a QR code"""
    generate_qr_code(data, filename)
    print(f"Data embedded in QR code: {filename}")

def extract_from_qr(filename="compressed_qr.png"):
    """Extract data from a QR code"""
    # This is a simplified version - in practice, you'd use a QR decoder library
    # For this example, we'll assume the QR code contains the compressed data as text
    try:
        with open(filename.replace('.png', '.txt'), 'r') as f:
            return f.read()
    except:
        print(f"QR code decoding not implemented. Using fallback from file.")
        with open("compressed_data_fallback.txt", 'r') as f:
            return f.read()

# Multi-level LZW compression
def multi_level_lzw_compress(data, levels=10):
    """Apply LZW compression multiple times"""
    compressed_data = data
    all_compressed = []
    
    for i in range(levels):
        print(f"\n--- Compression Level {i+1} ---")
        print(f"Input characters: {len(compressed_data)}")
        
        # LZW Compression
        compressed = lzw_compress(compressed_data)
        compressed_str = ' '.join(map(str, compressed))
        print(f"Compressed entries: {len(compressed)}")
        print(f"Compressed string length: {len(compressed_str)}")
        
        # Save intermediate results
        save_list(f"compressed_level_{i+1}.txt", compressed)
        save_text(f"compressed_str_level_{i+1}.txt", compressed_str)
        
        # Set up for next iteration
        compressed_data = compressed_str
        all_compressed.append(compressed)
    
    return compressed_data, all_compressed

# Multi-level LZW decompression
def multi_level_lzw_decompress(compressed_data, levels=10):
    """Decompress data that was compressed multiple times with LZW"""
    decompressed_data = compressed_data
    
    for i in range(levels-1, -1, -1):
        print(f"\n--- Decompression Level {i+1} ---")
        
        if i == levels-1:
            # For the last level, we have the compressed string
            # Convert back to list of integers
            compressed_list = list(map(int, decompressed_data.split()))
        else:
            # For intermediate levels, we have a string that needs to be converted to list
            compressed_list = list(map(int, decompressed_data.split()))
        
        print(f"Compressed entries to decompress: {len(compressed_list)}")
        
        # LZW Decompression
        decompressed_data = lzw_decompress(compressed_list)
        print(f"Decompressed characters: {len(decompressed_data)}")
    
    return decompressed_data

# Main pipeline
def qr_vc_pipeline(data):
    print("\n--- Starting QR-VC Pipeline ---")

    # Step 1: Generate and save QR code
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
    size = share1.shape
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

    # Step 4: Apply multi-level LZW compression
    compressed_data, all_compressed = multi_level_lzw_compress(flat_text, levels=10)
    
    # Save the final compressed data for fallback
    save_text("compressed_data_fallback.txt", compressed_data)
    
    # Save all compressed data
    save_json("all_compressed.json", all_compressed)
    
    # Comment out QR embedding (optional)
    '''
    # Step 5: Embed compressed data in QR code
    embed_in_qr(compressed_data, "compressed_qr.png")
    '''
    
    return size, compressed_data, all_compressed

# Run the compression pipeline
share_shape, compressed_data, all_compressed = qr_vc_pipeline("Transaction ID: 12345, Amount: $90")

# Get compressed data (from QR or fallback)
# Comment out QR extraction (optional)
'''
compressed_data_from_qr = extract_from_qr("compressed_qr.png")
'''
# Use fallback data if QR extraction is commented
compressed_data_from_file = compressed_data

# Decompress the data
print("\n--- Starting Multi-Level LZW Decompression ---")
decompressed_text = multi_level_lzw_decompress(compressed_data_from_file, levels=10)

# Reshape to image
recovered_array = np.array(list(map(int, decompressed_text)), dtype=np.uint8).reshape(share_shape)
save_image(recovered_array, "recovered_share1.png")
print("Recovered share saved as recovered_share1.png")

# Verify the decompression
# Read original share data
with open("share1_flattened.txt", "r") as f:
    original_text = f.read().strip()

# Compare
if original_text == decompressed_text:
    print("\nSUCCESS: Decompressed data matches original exactly!")
    print(f"Original size: {len(original_text)} characters")
    print(f"Decompressed size: {len(decompressed_text)} characters")
else:
    print("\nERROR: Decompressed data does not match original!")
    # Count differences
    differences = sum(1 for a, b in zip(original_text, decompressed_text) if a != b)
    print(f"Number of differing characters: {differences}")
    print(f"Match percentage: {(1 - differences/len(original_text)) * 100:.2f}%")