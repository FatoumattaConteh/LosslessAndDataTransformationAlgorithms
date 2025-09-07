import qrcode
import random
import numpy as np
from PIL import Image
import heapq
from collections import Counter, defaultdict

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

# RLE Compression
def rle_compress(data):
    if not data:
        return ""
    compressed = []
    prev_char = data[0]
    count = 1
    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            compressed.append(f"{count}{prev_char}")
            prev_char = char
            count = 1
    compressed.append(f"{count}{prev_char}")
    return ''.join(compressed)

# RLE Decompression
def rle_decompress(data):
    import re
    matches = re.findall(r'(\d+)(\D)', data)
    decompressed = ''.join([char * int(count) for count, char in matches])
    return decompressed

# Save helpers
def save_image(array, filename):
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(filename, format="PNG")

def save_text(filename, data):
    with open(filename, 'w') as f:
        f.write(data)
    print(f"{filename} saved with {len(data)} characters.")

def save_list(filename, data):
    with open(filename, 'w') as f:
        f.write(' '.join(map(str, data)))
    print(f"{filename} saved with {len(data)} entries.")

# Huffman for Integer Lists
class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree_int(data):
    freq = Counter(data)
    heap = [HuffmanNode(val, freq[val]) for val in freq]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def build_codes_int(node, prefix="", code_map=None):
    if code_map is None:
        code_map = dict()
    if node:
        if node.value is not None:
            code_map[node.value] = prefix
        build_codes_int(node.left, prefix + "0", code_map)
        build_codes_int(node.right, prefix + "1", code_map)
    return code_map

def huffman_compress_int(data):
    root = build_huffman_tree_int(data)
    code_map = build_codes_int(root)
    compressed = ''.join(code_map[val] for val in data)
    return compressed, code_map

def huffman_decompress_int(bitstring, code_map):
    reverse_map = {v: k for k, v in code_map.items()}
    buffer = ""
    output = []

    for bit in bitstring:
        buffer += bit
        if buffer in reverse_map:
            output.append(reverse_map[buffer])
            buffer = ""
    return output


# Main pipeline
def qr_vc_pipeline(data):
    print("\n--- Starting QR-VC Pipeline ---")

    # Step 1: Generate and save QR code
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr_code(data, "original_qr.png")

    # Step 2: Create Shares
    share1, share2 = create_shares(qr_array)
    save_image(share1, "share1.png")
    save_image(share2, "share2.png")
    print(f"Size of Share 1: {share1.shape}")

    # Step 3: Flatten Share1 and Save as Text
    flat_share1 = share1.flatten()
    flat_text = ''.join(map(str, flat_share1))
    save_text("share1_flattened.txt", flat_text)
    print(f"Characters in Original Share 1: {len(flat_text)}")

        # Step 4: LZW Compression
    compressed_lzw = lzw_compress(flat_text)
    save_list("share1_lzw.txt", compressed_lzw)
    print(f"Integer chunks after LZW: {len(compressed_lzw)}")

    # Step 5: Huffman Encoding of LZW output
    compressed_huffman, huff_map = huffman_compress_int(compressed_lzw)
    save_text("share1_lzw_huffman.txt", compressed_huffman)
    print(f"Bits after LZW + Huffman: {len(compressed_huffman)}")

    # Step 6: Huffman Decoding
    decompressed_lzw = huffman_decompress_int(compressed_huffman, huff_map)
    print(f"Recovered LZW codes: {len(decompressed_lzw)}")

    # Step 7: LZW Decompression
    decompressed_str = lzw_decompress(decompressed_lzw)
    save_text("share1_decompressed.txt", decompressed_str)
    print(f"Characters in Final Decompressed: {len(decompressed_str)}")


        # Step 4: RLE Compression
    compressed_rle = rle_compress(flat_text)
    save_text("share1_rle.txt", compressed_rle)
    print(f"Characters after RLE Compression: {len(compressed_rle)}")


    # Step 8: Reconstruct Share 1
    recovered = np.array(list(map(int, decompressed_str)), dtype=np.uint8).reshape(share1.shape)
    print(f"Decompressed Share 1 matches original: {np.array_equal(share1, recovered)}")


# Run the pipeline
qr_vc_pipeline("Hello from QR-VC!")
