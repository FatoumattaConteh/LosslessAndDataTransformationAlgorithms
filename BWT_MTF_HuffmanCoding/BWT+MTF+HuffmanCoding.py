import qrcode
import random
import numpy as np
from PIL import Image
import heapq
from collections import defaultdict, Counter
import json
import base64
import os
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

# BWT transform
def bwt_transform(s):
    s += "$"
    suffixes = [(s[i:], i) for i in range(len(s))]
    suffixes.sort()
    bwt_result = ''.join(s[i - 1] if i != 0 else "$" for _, i in suffixes)
    return bwt_result

# Save helper functions
def save_image(array, filename):
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(filename, format="PNG")

def save_text(filename, data):
    with open(filename, 'w') as f:
        f.write(data)
    print(f"{filename} saved with {len(data)} characters.")

def mtf_encode(data):
    # Initialize the alphabet from the unique symbols in the data
    alphabet = list(dict.fromkeys(sorted(data)))  # preserves order, no duplicates
    output = []
    for char in data:
        index = alphabet.index(char)
        output.append(index)
        # Move the accessed character to front
        alphabet.insert(0, alphabet.pop(index))
    return output, alphabet


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(root):
    codes = {}
    def generate_code(node, code):
        if node:
            if node.char is not None:
                codes[node.char] = code
            generate_code(node.left, code + '0')
            generate_code(node.right, code + '1')
    generate_code(root, '')
    return codes

def huffman_encode(data):
    freq_dict = Counter(data)
    root = build_huffman_tree(freq_dict)
    codes = build_huffman_codes(root)
    encoded = ''.join(codes[char] for char in data)
    return encoded, codes

# ------------------- DECODING FUNCTIONS -------------------

def huffman_decode(encoded, codes):
    # Build reverse dictionary
    reverse_codes = {v: k for k, v in codes.items()}
    decoded = []
    buffer = ""
    for bit in encoded:
        buffer += bit
        if buffer in reverse_codes:
            decoded.append(reverse_codes[buffer])
            buffer = ""
    return decoded

def mtf_decode(encoded, alphabet):
    decoded = []
    for index in encoded:
        index = int(index)  # ensure integer
        symbol = alphabet[index]
        decoded.append(symbol)
        # Move symbol to front
        alphabet.insert(0, alphabet.pop(index))
    return decoded

def bwt_inverse(bwt_result):
    n = len(bwt_result)
    table = [""] * n
    for _ in range(n):
        table = sorted([bwt_result[i] + table[i] for i in range(n)])
    # Find row that ends with "$"
    for row in table:
        if row.endswith("$"):
            return row.rstrip("$")

# ------------------- SAVE/LOAD HELPERS -------------------
def save_json(filename, obj):
    with open(filename, "w") as f:
        json.dump(obj, f)

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

# ------------------- QR CODE EMBEDDING AND EXTRACTION -------------------
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

# ------------------- MAIN PIPELINE -------------------
def qr_vc_pipeline(data):
    print("\n--- Starting QR-VC Pipeline ---")

    # Step 1: Generate and save QR code
    qr_array = generate_qr_code(data, "original_qr.png")

    ### ADDED: Original QR entropy analysis
    qr_binary = "".join("1" if bit else "0" for row in qr_array for bit in row)
    qr_zeros = qr_binary.count("0")
    qr_ones = qr_binary.count("1")
    qr_entropy = shannon_entropy(qr_binary)
    print("\n=== Original QR Code Binary ===")
    print(f"Size: {len(qr_binary)} bits")
    print(f"Number of 0s: {qr_zeros}, Number of 1s: {qr_ones}")
    print(f"Shannon Entropy: {qr_entropy:.4f} bits/symbol\n")

    # Step 2: Create shares
    share1, share2 = create_shares(qr_array)
    save_image(share1, "share1.png")
    save_image(share2, "share2.png")
    size = share1.shape
    print(f"Size of share1: {size}")


    # Step 3: Flatten share1 and save as text
    flat_share1 = share1.flatten()
    flat_text = ''.join(map(str, flat_share1))
    save_text("share1_flattened.txt", flat_text)
    print(f"Characters in original Share 1: {len(flat_text)}")


    ### ADDED: Count zeros and ones
    zeros = flat_text.count("0")
    ones = flat_text.count("1")
    print(f"Number of 0s: {zeros}, Number of 1s: {ones}")

    ### ADDED: Shannon Entropy
    entropy_value = shannon_entropy(flat_text)
    print(f"Shannon Entropy (before compression): {entropy_value:.4f} bits/symbol")

    # Step 4: BWT Transform
    bwt_result = bwt_transform(flat_text)
    save_text("share1_bwt.txt", bwt_result)
    print(f"Characters in BWT Compressed Share 1: {len(bwt_result)}")

    # Step 5: MTF Encoding after BWT
    mtf_result, mtf_alphabet = mtf_encode(bwt_result)
    mtf_text = ' '.join(map(str, mtf_result))
    save_text("share1_bwt_mtf.txt", mtf_text)
    print(f"Characters in BWT+MTF Encoded Share 1: {len(mtf_text)}")

    # Step 6: Huffman Encoding after MTF
    mtf_chars = list(map(str, mtf_result))  # Convert ints to strings
    huffman_encoded, huff_codes = huffman_encode(mtf_chars)
    save_text("share1_bwt_mtf_huffman.txt", huffman_encoded)
    print(f"Bits in Huffman Encoded Share 1 (BWT + MTF): {len(huffman_encoded)}")

    # Save Huffman codes + alphabet for decoding
    save_json("huffman_codes.json", huff_codes)
    save_json("mtf_alphabet.json", mtf_alphabet)
    
    # Save the compressed data for fallback
    save_text("compressed_data_fallback.txt", huffman_encoded)
    
    # Comment out QR embedding (optional)
    '''
    # Step 7: Embed compressed data in QR code
    embed_in_qr(huffman_encoded, "compressed_qr.png")
    '''

    return size, huffman_encoded


def decode_qr_vc(compressed_data, codes_file, alphabet_file, share_shape):
    print("\n--- Starting QR-VC Decoding ---")

    # Step 1: Load codes + alphabet
    huff_codes = load_json(codes_file)
    alphabet = load_json(alphabet_file)

    # Step 2: Huffman Decoding
    mtf_chars = huffman_decode(compressed_data, huff_codes)
    print(f"Recovered {len(mtf_chars)} MTF symbols")

    # Step 3: Inverse MTF
    mtf_decoded = mtf_decode(mtf_chars, alphabet)
    bwt_lastcol = ''.join(mtf_decoded)
    print("Recovered BWT last column")

    # Step 4: Inverse BWT
    recovered_text = bwt_inverse(bwt_lastcol)
    print(f"Recovered original share text, length = {len(recovered_text)}")

    # Step 5: Reshape to image
    recovered_array = np.array(list(map(int, recovered_text)), dtype=np.uint8).reshape(share_shape)
    save_image(recovered_array, "recovered_share1.png")
    print("Recovered share saved as recovered_share1.png")

    return recovered_array, recovered_text


# Run it
share_shape, compressed_data = qr_vc_pipeline("Transaction ID: 12345, Amount: $90")

# Get compressed data (from QR or fallback)
# Comment out QR extraction (optional)
'''
compressed_data_from_qr = extract_from_qr("compressed_qr.png")
'''
# Use fallback data if QR extraction is commented
compressed_data_from_file = compressed_data

# Decode the data
recovered_array, recovered_text = decode_qr_vc(
    compressed_data=compressed_data_from_file,
    codes_file="huffman_codes.json",
    alphabet_file="mtf_alphabet.json",
    share_shape=share_shape,
)

# Verify the decompression
# Read original share data
with open("share1_flattened.txt", "r") as f:
    original_text = f.read().strip()

# Compare
if original_text == recovered_text:
    print("SUCCESS: Decompressed data matches original exactly!")
    print(f"Original size: {len(original_text)} characters")
    print(f"Decompressed size: {len(recovered_text)} characters")
else:
    print("ERROR: Decompressed data does not match original!")
    # Count differences
    differences = sum(1 for a, b in zip(original_text, recovered_text) if a != b)
    print(f"Number of differing characters: {differences}")
    print(f"Match percentage: {(1 - differences/len(original_text)) * 100:.2f}%")