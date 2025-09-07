import qrcode
import numpy as np
import random 
from PIL import Image
import json
import cv2
import os
import struct

# Generate QR Code
def generate_qr(data, filename="qr_code.png"):
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
            if qr_array[i, j] == 0:
                pattern = random.choice([[0, 0], [1, 1]])
            else:
                pattern = random.choice([[0, 1], [1, 0]])
            share1[i, j] = pattern[0]
            share2[i, j] = pattern[1]
    return share1, share2

# Save share as image
def save_share_as_image(share, filename):
    img = Image.fromarray((share * 255).astype(np.uint8))
    img.save(filename, format="PNG")

# RLE Encoding
def run_length_encoding_optimized(data):
    if len(data) == 0:
        return []
    
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append(data[i - 1])
            encoded.append(count)
            count = 1
    encoded.append(data[-1])
    encoded.append(count)
    return encoded

# RLE Decoding
def run_length_decoding_optimized(encoded):
    if not encoded:
        return []
    
    decoded = []
    for i in range(0, len(encoded), 2):
        if i + 1 < len(encoded):
            value = encoded[i]
            count = encoded[i + 1]
            decoded.extend([value] * count)
    return decoded

# Save vector to text
def save_vector_to_text(vector, filename):
    with open(filename, 'w') as file:
        file.write(','.join(map(str, vector)))

# Save vector to binary
def save_vector_to_binary(vector, filename):
    with open(filename, 'wb') as file:
        for val in vector:
            file.write(struct.pack('B', val))

# Calculate memory size
def calculate_memory_size(vector):
    return len(vector) * struct.calcsize('B')

# Reconstruct QR - FIXED THE TYPO HERE
def reconstruct_qr(share1, share2, threshold=127):
    reconstructed = np.bitwise_xor(share1, share2).astype(np.uint8) * 255
    return (reconstructed > threshold).astype(np.uint8)

# Save data to file
def save_data_to_file(data, filename):
    with open(filename, 'w') as file:
        file.write(data)
    print(f"Data saved to {filename}")

# Read data from file
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# QR Code embedding
def embed_in_qr(data, filename="compressed_qr.png"):
    """Embed data in a QR code"""
    generate_qr(data, filename)
    print(f"Data embedded in QR code: {filename}")

# QR Code extraction
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

# Multi-level RLE compression
def multi_level_rle_compress(data, levels=10):
    """Apply RLE compression multiple times"""
    compressed_data = data
    all_compressed = []
    
    for i in range(levels):
        print(f"\n--- Compression Level {i+1} ---")
        print(f"Input elements: {len(compressed_data)}")
        
        # RLE Compression
        compressed = run_length_encoding_optimized(compressed_data)
        compressed_str = ','.join(map(str, compressed))
        print(f"Compressed elements: {len(compressed)}")
        print(f"Compressed string length: {len(compressed_str)}")
        
        # Save intermediate results
        save_vector_to_text(compressed, f"compressed_level_{i+1}.txt")
        save_data_to_file(compressed_str, f"compressed_str_level_{i+1}.txt")
        
        # Set up for next iteration
        compressed_data = compressed
        all_compressed.append(compressed)
    
    return compressed_data, all_compressed

# Multi-level RLE decompression
def multi_level_rle_decompress(compressed_data, levels=10):
    """Decompress data that was compressed multiple times with RLE"""
    decompressed_data = compressed_data
    
    for i in range(levels-1, -1, -1):
        print(f"\n--- Decompression Level {i+1} ---")
        print(f"Compressed elements to decompress: {len(decompressed_data)}")
        
        # RLE Decompression
        decompressed_data = run_length_decoding_optimized(decompressed_data)
        print(f"Decompressed elements: {len(decompressed_data)}")
    
    return decompressed_data

# Main pipeline
def qr_vc_pipeline(data):
    print("\n--- Starting QR-VC Pipeline ---")

    # Step 1: Generate and save QR code
    qr_array = generate_qr(data, "original_qr.png")

    # Step 2: Create shares
    share1, share2 = create_shares(qr_array)
    save_share_as_image(share1, "share1.png")
    save_share_as_image(share2, "share2.png")
    size = share1.shape
    print(f"Size of share1: {size}")

    # Step 3: Flatten share1 and save as text
    flat_share1 = share1.ravel()
    flat_list = flat_share1.tolist()
    flat_text = ''.join(map(str, flat_share1))
    save_data_to_file(flat_text, "multipleOptimized_share1_flattened.txt")
    print(f"Elements in original Share 1: {len(flat_list)}")

    # Step 4: Apply multi-level RLE compression
    compressed_data, all_compressed = multi_level_rle_compress(flat_list, levels=10)
    
    # Save the final compressed data for fallback
    compressed_str = ','.join(map(str, compressed_data))
    save_data_to_file(compressed_str, "multiple_optimized_compressed_data_fallback.txt")
    
    # Save all compressed data
    with open("all_compressed.json", "w") as f:
        json.dump(all_compressed, f)
    
    # Comment out QR embedding (optional)
    '''
    # Step 5: Embed compressed data in QR code
    embed_in_qr(compressed_str, "compressed_qr.png")
    '''
    
    return size, compressed_data, all_compressed, share2

# Run the compression pipeline
share_shape, compressed_data, all_compressed, share2 = qr_vc_pipeline("Transaction ID: 12345, Amount: $90")

# Get compressed data (from QR or fallback)
# Comment out QR extraction (optional)
'''
compressed_str_from_qr = extract_from_qr("compressed_qr.png")
compressed_data_from_qr = list(map(int, compressed_str_from_qr.split(',')))
'''
# Use fallback data if QR extraction is commented
compressed_data_from_file = compressed_data

# Decompress the data
print("\n--- Starting Multi-Level RLE Decompression ---")
decompressed_list = multi_level_rle_decompress(compressed_data_from_file, levels=10)

# Check if the decompressed data has the correct length
expected_size = share_shape[0] * share_shape[1]
if len(decompressed_list) != expected_size:
    print(f"Warning: Decompressed size ({len(decompressed_list)}) doesn't match expected size ({expected_size})")
    # Pad or truncate to match expected size
    if len(decompressed_list) < expected_size:
        decompressed_list.extend([0] * (expected_size - len(decompressed_list)))
    else:
        decompressed_list = decompressed_list[:expected_size]

# Reshape to image
recovered_array = np.array(decompressed_list, dtype=np.uint8).reshape(share_shape)
save_share_as_image(recovered_array, "recovered_share1.png")
print("Recovered share saved as recovered_share1.png")

# Verify the decompression
# Read original share data
with open("share1_flattened.txt", "r") as f:
    original_text = f.read().strip()

# Convert to list for comparison
original_list = [int(char) for char in original_text]
decompressed_text = ''.join(map(str, decompressed_list))

# Compare
if original_list == decompressed_list:
    print("\nSUCCESS: Decompressed data matches original exactly!")
    print(f"Original size: {len(original_list)} elements")
    print(f"Decompressed size: {len(decompressed_list)} elements")
else:
    print("\nERROR: Decompressed data does not match original!")
    # Count differences
    differences = sum(1 for a, b in zip(original_list, decompressed_list) if a != b)
    print(f"Number of differing elements: {differences}")
    print(f"Match percentage: {(1 - differences/len(original_list)) * 100:.2f}%")

# Reconstruct QR from shares
reconstructed_qr = reconstruct_qr(recovered_array, share2)
save_share_as_image(reconstructed_qr, "reconstructed_qr_from_share1.png")

# Verify QR reconstruction
original_qr = generate_qr("Transaction ID: 12345, Amount: $90", "original_qr.png")
if np.array_equal(original_qr, reconstructed_qr):
    print("The QR reconstruction was successful! The QR code matches the original.")
else:
    print("QR reconstruction failed: The QR code does not match the original.")