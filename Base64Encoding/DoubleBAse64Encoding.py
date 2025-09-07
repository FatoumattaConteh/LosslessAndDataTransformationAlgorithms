import numpy as np
import qrcode
import random
import json
from PIL import Image
import base64
import os
import math   ### ADDED

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

# Step 3: Base64 Encoding
def base64_encode(data):
    return base64.b64encode(data).decode("utf-8")

# Step 4: Save Data to File
def save_data_to_file(data, filename):
    with open(filename, "w") as file:
        file.write(data)
    print(f"Data saved to {filename}")

# Read data from file
def read_data_from_file(filename):
    with open(filename, "r") as file:
        return file.read()

# Step 6: Reconstruct the QR Code
def reconstruct_qr(share1, share2):
    rows, cols = share1.shape
    reconstructed_qr = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if share1[i, j] == share2[i, j]:
                reconstructed_qr[i, j] = 0  # Black
            else:
                reconstructed_qr[i, j] = 1  # White

    return reconstructed_qr

# Main Workflow
if __name__ == "__main__":
    # Step 1: Data for the QR code
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


    # Step 2: Create visual cryptography shares
    share1, share2 = create_shares(qr_array)

    # Convert share1 to a binary string
    share1_binary = "".join(map(str, share1.flatten()))
    
    # Print character count before compression
    print(f"Characters before compression: {len(share1_binary)}")
    
    # Group bits into bytes before Base64 encoding
    share1_bytes = bytearray(int(share1_binary[i:i+8], 2) for i in range(0, len(share1_binary), 8))
    ### ADDED: Count zeros and ones
    zeros = share1_binary.count("0")
    ones = share1_binary.count("1")
    print(f"Number of 0s: {zeros}, Number of 1s: {ones}")

    ### ADDED: Shannon Entropy
    entropy_value = shannon_entropy(share1_binary)
    print(f"Shannon Entropy (before compression): {entropy_value:.4f} bits/symbol")

    if abs(entropy_value - 1.0) < 0.05:
        print("Data is HIGH entropy (random-like, less compressible).")
    elif entropy_value > 0.7:
        print("Data has MODERATE entropy (some patterns, partially compressible).")
    else:
        print("Data has LOW entropy (predictable, highly compressible).")

    # Save share1 as a binary text file before encoding
    save_data_to_file(share1_binary, "share1_before_encoding.txt")
    print("Share1 saved as binary text to 'share1_before_encoding.txt'")

    # Apply Base64 encoding 10 times
    encoded_data = share1_bytes
    for i in range(10):
        encoded_data = base64.b64encode(encoded_data)
        print(f"Characters after {i+1}th Base64 encoding: {len(encoded_data)}")
    
    # Save the final Base64 encoded share1 to a text file
    encoded_share1_str = encoded_data.decode("utf-8")
    save_data_to_file(encoded_share1_str, "encoded_share1.txt")
    print("Encoded share1 saved to 'encoded_share1.txt'")
    
    # Comment out QR embedding (optional)
    '''
    # Step 5: Embed the base64 encoded data into a QR code
    generate_qr(encoded_share1_str, "encoded_qr.png")
    print(f"Base64 encoded share1 embedded into 'encoded_qr.png'")
    '''

    # Read the encoded data from file (works regardless of QR embedding)
    print("Reading encoded data from file...")
    encoded_data_from_file = read_data_from_file("encoded_share1.txt").encode("utf-8")
    
    # Decode Base64 step by step until the original share1 bytes
    recovered_data = encoded_data_from_file
    for i in range(10):
        recovered_data = base64.b64decode(recovered_data)
        print(f"Characters after {10 - i}th Base64 decoding: {len(recovered_data)}")
    
    # Convert back to binary string
    recovered_binary = "".join(f"{byte:08b}" for byte in recovered_data)
    
    # Print character count after decompression
    print(f"Characters after decompression: {len(recovered_binary)}")
    
    # Convert binary string back to numpy array
    share1_reconstructed = np.array([int(bit) for bit in recovered_binary], dtype=np.uint8).reshape(share1.shape)
    
    # Verify if the decompressed data matches the original
    original_binary = ''.join(map(str, share1.flatten()))
    decompressed_binary = ''.join(map(str, share1_reconstructed.flatten()))
    
    if original_binary == decompressed_binary:
        print("SUCCESS: Decompressed data matches original exactly!")
        print(f"Original size: {len(original_binary)} characters")
        print(f"Decompressed size: {len(decompressed_binary)} characters")
    else:
        print("ERROR: Decompressed data does not match original!")
        # Count differences
        differences = sum(1 for a, b in zip(original_binary, decompressed_binary) if a != b)
        print(f"Number of differing characters: {differences}")
    
    # Step 7: Reconstruct the QR code from the shares
    reconstructed_qr = reconstruct_qr(share1_reconstructed, share2)

    # Save the reconstructed QR code
    Image.fromarray(reconstructed_qr * 255).save("reconstructed_qr.png")
    print("Reconstruction complete. Check 'reconstructed_qr.png' for the output.")