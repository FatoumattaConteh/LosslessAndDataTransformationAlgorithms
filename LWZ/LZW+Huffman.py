import qrcode
import random
import numpy as np
from PIL import Image
from collections import Counter
import heapq

# Generate QR Code
def generate_qr_code(data, version=1, box_size=10, border=4):
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size,
        border=border
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    return img

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

# Huffman Coding
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_map):
    heap = [HuffmanNode(sym, freq) for sym, freq in freq_map.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix='', codebook={}):
    if node:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        build_codes(node.left, prefix + '0', codebook)
        build_codes(node.right, prefix + '1', codebook)
    return codebook

def huffman_compress(data):
    freq_map = Counter(data)
    tree = build_huffman_tree(freq_map)
    codebook = build_codes(tree)
    encoded = ''.join(codebook[sym] for sym in data)
    return encoded, codebook

# Save helper functions
def save_image(array, filename):
    img = Image.fromarray(array * 255).convert("L")
    img.save(filename)

def save_text(filename, data):
    with open(filename, 'w') as f:
        f.write(data)
    print(f"{filename} saved with {len(data)} characters.")

def save_list(filename, data):
    text = ' '.join(map(str, data))
    with open(filename, 'w') as f:
        f.write(text)
    print(f"{filename} saved with {len(text)} characters.")

# Main pipeline
def qr_vc_pipeline(data):
    print("\n--- Starting QR-VC Pipeline ---")

    # Step 1: Generate and save QR code
    qr_img = generate_qr_code(data)
    qr_img.save("original_qr.png")
    qr_array = np.array(qr_img.convert("1"), dtype=np.uint8)

    # Step 2: Create shares
    share1, share2 = create_shares(qr_array)
    print(f"Size of share1: {share1.shape}")
    save_image(share1, "share1.png")
    save_image(share2, "share2.png")

    # Step 3: Flatten share1 and save
    flat_share1 = share1.flatten()
    flat_text = ''.join(map(str, flat_share1))
    save_text("share1_flattened.txt", flat_text)
    print(f"Characters in original Share 1: {len(flat_text)}")

    # Step 4: LZW Compression Round 1
    compressed1 = lzw_compress(flat_text)
    save_list("share1_lzw1.txt", compressed1)
    print(f"Characters in LZW Compressed Share 1 (Round 1): {len(' '.join(map(str, compressed1)))}")

    # Step 5: Huffman Encoding
    huffman_encoded, _ = huffman_compress(compressed1)
    save_text("share1_lzw1_huffman.txt", huffman_encoded)
    print(f"Characters in Huffman Compressed File: {len(huffman_encoded)}")

    # Step 6: Decompress back
    decompressed = lzw_decompress(compressed1.copy())
    save_text("share1_decompressed.txt", decompressed)
    print(f"Characters in Decompressed Share 1: {len(decompressed)}")

    # Step 7: Verify reconstruction
    recovered = np.array(list(map(int, decompressed)), dtype=np.uint8).reshape(share1.shape)
    print(f"Decompressed Share 1 matches original: {np.array_equal(share1, recovered)}")

# Run it
qr_vc_pipeline("Hello from QR-VC!")
