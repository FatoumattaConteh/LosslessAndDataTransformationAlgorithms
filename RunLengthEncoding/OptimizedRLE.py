import qrcode
import numpy as np
import random 
from PIL import Image
import json
import cv2
import os
import struct

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

def run_length_encoding_optimized(data):
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append(count)
            count = 1
    encoded.insert(0, data[0])
    encoded.append(count)
    return encoded

def run_length_decoding_optimized(encoded):
    if not encoded:
        return []
    first_value = encoded[0]
    decoded = []
    current_value = first_value
    for count in encoded[1:]:
        decoded.extend([current_value] * count)
        current_value = 1 - current_value
    return decoded

def save_vector_to_text(vector, filename):
    with open(filename, 'w') as file:
        file.write(','.join(map(str, vector)))

def save_vector_to_binary(vector, filename):
    with open(filename, 'wb') as file:
        file.write(struct.pack('B', vector[0]))
        for count in vector[1:]:
            file.write(struct.pack('B', count))

def calculate_memory_size(vector):
    return struct.calcsize('B') + len(vector[1:]) * struct.calcsize('B')

def reconstruct_qr(share1, share2, threshold=127):
    reconstructed = np.bitwise_xor(share1, share2).astype(np.uint8) * 255
    return (reconstructed > threshold).astype(np.uint8)

if __name__ == "__main__":
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")
    share1, share2 = create_shares(qr_array)
    save_share_as_image(share1, "share1.png")
    save_share_as_image(share2, "share2.png")
    
    vector1 = share1.ravel()
    original_size = len(vector1)
    print(f"Size of share1 before compression: {original_size} elements")

    encoded_vector1 = run_length_encoding_optimized(vector1)
    print(f"Optimized RLE Encoded Vector Size: {len(encoded_vector1)} elements")
    
    memory_size = calculate_memory_size(encoded_vector1)
    print(f"Memory size of optimized RLE encoded vector: {memory_size * 8} bits")
    
    save_vector_to_text(encoded_vector1, "Optimized_encoded_vector1.txt")
    save_vector_to_binary(encoded_vector1, "Optimized_encoded_vector1.bin")
    
    decoded_vector1 = run_length_decoding_optimized(encoded_vector1)
    share1_restored = np.array(decoded_vector1).reshape(share1.shape)
    reconstructed_qr = reconstruct_qr(share1_restored, share2)
    save_share_as_image(reconstructed_qr, "reconstructed_qr_from_share1.png")

    if np.array_equal(qr_array, reconstructed_qr):
        print("The reconstruction was successful! The QR code matches the original.")
    else:
        print("Reconstruction failed: The QR code does not match the original.")
