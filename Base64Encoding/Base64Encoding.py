import numpy as np
import qrcode
import random
import base64
from PIL import Image
import cv2
from pyzbar.pyzbar import decode
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

# Convert binary string to bytes
def binary_to_bytes(binary_string):
    byte_array = bytearray()
    for i in range(0, len(binary_string), 8):
        byte_chunk = binary_string[i:i+8]
        if len(byte_chunk) < 8:
            byte_chunk = byte_chunk.ljust(8, '0')
        byte_array.append(int(byte_chunk, 2))
    return bytes(byte_array)

# Convert bytes to binary string
def bytes_to_binary_string(byte_data):
    return ''.join(format(byte, '08b') for byte in byte_data)

# Base64 encode the byte data
def base64_encode_binary(binary_string):
    byte_data = binary_to_bytes(binary_string)
    return base64.b64encode(byte_data).decode("utf-8")

# Base64 decode back to binary string
def base64_decode_to_binary(encoded_string):
    byte_data = base64.b64decode(encoded_string)
    return bytes_to_binary_string(byte_data)

# Step 4: Save Data to File
def save_data_to_file(data, filename):
    with open(filename, "w") as file:
        file.write(data)
    print(f"Data saved to {filename}")

# Step 5: Decode QR Code to extract Base64 string
def decode_qr_image(filename):
    img = cv2.imread(filename)
    decoded_objects = decode(img)
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8')
    else:
        raise ValueError("No QR code detected in the image.")

# Step 6: Reconstruct the QR Code
def reconstruct_qr(share1, share2):
    rows, cols = share1.shape
    reconstructed_qr = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if share1[i, j] == share2[i, j]:
                reconstructed_qr[i, j] = 0
            else:
                reconstructed_qr[i, j] = 1
    return reconstructed_qr

# Read data from file
def read_data_from_file(filename):
    with open(filename, "r") as file:
        return file.read()

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

    # Convert share1 to binary string
    share1_binary = "".join(map(str, share1.flatten()))
    
    # Print size before compression
    print("=== VC Share 1 Binary ===")
    print(f"Size before compression: {len(share1_binary)} bits")
    print(f"Characters before compression: {len(share1_binary)}")

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

    # Save before encoding
    save_data_to_file(share1_binary, "share1_before_encoding.txt")

    # Base64 encode
    encoded_share1 = base64_encode_binary(share1_binary)
    
    # Print size after compression
    print(f"Characters after Base64 encoding: {len(encoded_share1)}")
    save_data_to_file(encoded_share1, "encoded_share1.txt")

    # Comment out QR embedding (optional)
    '''
    # Embed into QR
    generate_qr(encoded_share1, "encoded_qr.png")
    print("Base64 encoded share1 embedded into 'encoded_qr.png'")

    # Step 3: Decode Base64 back from QR image
    print("Decoding Base64 from QR image...")
    extracted_base64 = decode_qr_image("encoded_qr.png")
    '''
    
    # Alternative: Read from file instead of QR
    print("Reading Base64 from file...")
    extracted_base64 = read_data_from_file("encoded_share1.txt")

    # Step 4: Decode Base64 to binary string
    decoded_binary = base64_decode_to_binary(extracted_base64)
    
    # Print size after decompression
    print(f"Characters after decompression: {len(decoded_binary)}")

    # Ensure the binary string fits the shape
    expected_size = share1.shape[0] * share1.shape[1]
    decoded_binary = decoded_binary[:expected_size]  # truncate if padded

    # Convert back to array
    decoded_share1 = np.array(list(decoded_binary), dtype=np.uint8).reshape(share1.shape)

    # Step 5: Reconstruct QR from shares
    reconstructed_qr = reconstruct_qr(decoded_share1, share2)
    Image.fromarray(reconstructed_qr * 255).save("reconstructed_qr.png")
    
    # Verify if the decompressed data matches the original
    original_binary = ''.join(map(str, share1.flatten()))
    decompressed_binary = ''.join(map(str, decoded_share1.flatten()))
    
    if original_binary == decompressed_binary:
        print("SUCCESS: Decompressed data matches original exactly!")
        print(f"Original size: {len(original_binary)} bits")
        print(f"Decompressed size: {len(decompressed_binary)} bits")
    else:
        print("ERROR: Decompressed data does not match original!")
        # Count differences
        differences = sum(1 for a, b in zip(original_binary, decompressed_binary) if a != b)
        print(f"Number of differing bits: {differences}")
    
    print("Reconstruction complete. Check 'reconstructed_qr.png'")

