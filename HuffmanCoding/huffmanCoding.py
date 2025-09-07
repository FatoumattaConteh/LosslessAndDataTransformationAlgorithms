import qrcode
import numpy as np
import random
from PIL import Image
import json
import cv2
import os
import heapq
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



class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

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

# Save share as a text file
def save_share_as_text(share, filename):
    with open(filename, 'w') as file:
        for row in share:
            file.write(''.join(str(int(pixel)) for pixel in row) + '\n')

# Huffman Coding Functions
def calculate_frequencies(data):
    freq = {}
    for value in data:
        freq[value] = freq.get(value, 0) + 1
    return freq

# Build Huffman Tree
def build_huffman_tree(freq_dict):
    priority_queue = [HuffmanNode(symbol, freq) for symbol, freq in freq_dict.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0] if priority_queue else None

# Generate Huffman Codes
def generate_huffman_codes(node, prefix="", code_table=None):
    if code_table is None:
        code_table = {}
    if node is not None:
        if node.symbol is not None:  # Leaf node
            code_table[int(node.symbol)] = prefix
        generate_huffman_codes(node.left, prefix + "0", code_table)
        generate_huffman_codes(node.right, prefix + "1", code_table)
    return code_table

# Encode Data Using Huffman Codes
def huffman_encode(data, code_table):
    return "".join(code_table[int(value)] for value in data)

# Decode Data Using Huffman Codes
def huffman_decode(encoded_data, code_table):
    # Create reverse mapping from code to symbol
    reverse_code_table = {v: k for k, v in code_table.items()}
    
    decoded_data = []
    current_code = ""
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_code_table:
            decoded_data.append(reverse_code_table[current_code])
            current_code = ""
    
    return decoded_data

# Reconstruct QR from shares
def reconstruct_qr(share1, share2):
    return np.bitwise_xor(share1, share2)

# Extract data from QR code
def extract_from_qr(filename):
    # In a real implementation, you would use a QR code decoder
    # For this example, we'll read from the saved text file
    try:
        with open(filename.replace('.png', '.txt'), 'r') as f:
            return f.read()
    except:
        print(f"QR code decoding not implemented. Using fallback from file.")
        with open("huffman_encoded_data.json", 'r') as f:
            data = json.load(f)
            return data["encoded_data"]

# Main workflow with Huffman coding integration
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
    save_share_as_image(share1, "share1.png")
    save_share_as_image(share2, "share2.png")

    # Step 3: Convert Share 1 to a vector and apply Huffman encoding
    vector1 = share1.ravel()  # Flatten Share 1
    count_0s_before = np.count_nonzero(vector1 == 0)
    count_1s_before = np.count_nonzero(vector1 == 1)
    print(f"Number of 0s in share1 before compression: {count_0s_before}")
    print(f"Number of 1s in share1 before compression: {count_1s_before}")

    
    # ---- Entropy after VC ----
    entropy_share1 = shannon_entropy(share1)
    num_zeros_s1 = np.count_nonzero(share1 == 0)
    num_ones_s1 = np.count_nonzero(share1 == 1)
    print("\n--- Share1 Statistics ---")
    print(f"Share1 size: {share1.size} bits")
    print(f"Number of 0s: {num_zeros_s1}, Number of 1s: {num_ones_s1}")
    print(f"Shannon Entropy (Share1): {entropy_share1:.4f} bits/symbol")


    freq_dict = calculate_frequencies(vector1)
    huffman_tree = build_huffman_tree(freq_dict)
    huffman_code_table = generate_huffman_codes(huffman_tree)
    compressed_data = huffman_encode(vector1, huffman_code_table)

    # Save compressed data to text file
    with open("share1_after_compression.txt", 'w') as file:
        file.write(compressed_data)

    # Display number of characters and 0s/1s in the compressed file
    num_characters = len(compressed_data)
    count_0s_after = compressed_data.count('0')
    count_1s_after = compressed_data.count('1')
    print(f"Number of characters in the compressed file: {num_characters}")
    print(f"Number of 0s in share1 after compression: {count_0s_after}")
    print(f"Number of 1s in share1 after compression: {count_1s_after}")

    # Step 4: Save Huffman code table and encoded data
    compressed_dict = {"code_table": {str(k): v for k, v in huffman_code_table.items()},
                       "encoded_data": compressed_data,
                       "shape": share1.shape}

    with open("huffman_encoded_data.json", 'w') as file:
        json.dump(compressed_dict, file, separators=(',', ':'))

    # Comment out QR embedding (optional)
    '''
    # Step 5: Embed the Huffman encoded data into a QR code
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(compressed_data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save("qr_huffman_share1.png")
    '''

    # DECOMPRESSION SECTION
    print("\n--- Starting Decompression ---")
    
    # Load the Huffman code table and shape
    with open("huffman_encoded_data.json", 'r') as file:
        compressed_dict = json.load(file)
    
    # Get encoded data (from QR or from file)
    # Comment out QR extraction (optional)
    '''
    encoded_data = extract_from_qr("qr_huffman_share1.png")
    '''
    # Use fallback data if QR extraction is commented
    encoded_data = compressed_dict["encoded_data"]
    
    # Convert code table back to proper format
    code_table = {int(k): v for k, v in compressed_dict["code_table"].items()}
    shape = tuple(compressed_dict["shape"])
    
    # Decode the data
    decoded_vector = huffman_decode(encoded_data, code_table)
    
    # Reshape to original share shape
    decoded_share1 = np.array(decoded_vector, dtype=np.uint8).reshape(shape)
    
    # Reconstruct QR code from shares
    reconstructed_qr = reconstruct_qr(decoded_share1, share2)
    save_share_as_image(reconstructed_qr, "reconstructed_qr.png")
    
    # Verify the decompression
    if np.array_equal(share1, decoded_share1):
        print("SUCCESS: Decompressed share matches original exactly!")
        print(f"Original size: {len(vector1)} elements")
        print(f"Decompressed size: {len(decoded_vector)} elements")
    else:
        print("ERROR: Decompressed share does not match original!")
        # Count differences
        differences = np.sum(share1 != decoded_share1)
        print(f"Number of differing elements: {differences}")
    
    # Verify QR reconstruction
    if np.array_equal(qr_array, reconstructed_qr):
        print("The QR reconstruction was successful! The QR code matches the original.")
    else:
        print("QR reconstruction failed: The QR code does not match the original.")