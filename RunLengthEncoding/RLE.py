import qrcode
import numpy as np
import random 
from PIL import Image
import json
import cv2
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
            if qr_array[i, j] == 0:
                pattern = random.choice([[0, 0], [1, 1]])
            else:
                pattern = random.choice([[0, 1], [1, 0]])
            share1[i, j] = pattern[0]
            share2[i, j] = pattern[1]
    return share1, share2

def save_share_as_image(share, filename):
    img = Image.fromarray((share * 255).astype(np.uint8))
    img.save(filename, format="PNG")

def run_length_encoding(data):
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

def run_length_decoding(encoded):
    decoded = []
    for i in range(0, len(encoded), 2):
        value = encoded[i]
        count = encoded[i + 1]
        decoded.extend([value] * count)
    return decoded

def save_rle_to_text(encoded_vector, filename):
    encoded_string = ' '.join(map(str, encoded_vector))  # Keep spaces in file
    with open(filename, 'w') as file:
        file.write(encoded_string)
    return encoded_string

def calculate_file_size(filename):
    with open(filename, 'r') as file:
        content = file.read()
    character_count = len(content.replace(" ", ""))  # Exclude spaces
    return character_count, content

def embed_data_in_qr(data, filename="embedded_qr.png"):
    qr = qrcode.QRCode(version=10, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)

def extract_data_from_qr(filename):
    qr_image = cv2.imread(filename)
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(qr_image)
    return data

def reconstruct_qr(share1, share2, threshold=127):
    reconstructed = np.bitwise_xor(share1, share2).astype(np.uint8) * 255
    return (reconstructed > threshold).astype(np.uint8)

if __name__ == "__main__":
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")

     # --- Entropy after QR ---
    qr_flat = qr_array.ravel().tolist()
    qr_entropy = shannon_entropy(qr_flat)
    print(f"[QR] Size: {len(qr_flat)} bits | Entropy: {qr_entropy:.4f}")


    share1, share2 = create_shares(qr_array)
    save_share_as_image(share1, "share1.png")
    save_share_as_image(share2, "share2.png")


    # --- Entropy after VC ---
    share1_flat = share1.ravel().tolist()
    vc_entropy = shannon_entropy(share1_flat)
    print(f"[VC Share1] Size: {len(share1_flat)} bits | Entropy: {vc_entropy:.4f}")
    
    vector1 = share1.ravel()
    original_size = len(vector1)
    print(f"Size of share1 before compression: {original_size} elements")

    encoded_vector1 = run_length_encoding(vector1)
    print(f"RLE Encoded Vector Size: {len(encoded_vector1) // 2} pairs")
    
    encoded_string = save_rle_to_text(encoded_vector1, "encoded_vector1.txt")
    character_count, file_content = calculate_file_size("encoded_vector1.txt")
    print(f"Number of characters in the saved file (excluding spaces): {character_count}")
    print("Compressed file content:")
    #print(file_content)

    # Save compressed RLE to file
    encoded_string = save_rle_to_text(encoded_vector1, "encoded_vector1.txt")

    # Generate a QR code that only stores the filename (or a small key)
    reference_data = "encoded_vector1.txt"  # Could also be a URL or identifier
    #embed_data_in_qr(reference_data, "embedded_reference_qr.png")

    # Later, for decompression:
    with open("encoded_vector1.txt", "r") as f:
        file_content = f.read()

    
    #embedded_qr = embed_data_in_qr(file_content, "embedded_qr.png")
    #extracted_data = extract_data_from_qr("embedded_qr.png")
    
    decoded_vector1 = run_length_decoding([int(x) for x in file_content.split()])
    share1_restored = np.array(decoded_vector1).reshape(share1.shape)
    reconstructed_qr = reconstruct_qr(share1_restored, share2)
    save_share_as_image(reconstructed_qr, "reconstructed_qr_from_share1.png")

    if np.array_equal(qr_array, reconstructed_qr):
        print("The reconstruction was successful! The QR code matches the original.")
    else:
        print("Reconstruction failed: The QR code does not match the original.")
