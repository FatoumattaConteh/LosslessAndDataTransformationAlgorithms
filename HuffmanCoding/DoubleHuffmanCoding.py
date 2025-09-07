import qrcode
import numpy as np
import random
from PIL import Image
import json
import os
import heapq
import ast
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


# Class to represent a node in the Huffman tree
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Function to generate a QR code
def generate_qr(data, filename="qr_code.png"):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)

# Function to create visual cryptography shares
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

# === REPLACEMENT / FIXED FUNCTIONS ===

def save_to_file(data, filename, binary=False):
    """
    If binary=False we save the exact bitstring as text (no padding).
    If binary=True we still write bytes, BUT we also write a .len file containing original bit length.
    """
    if binary:
        # write bytes
        byte_len = (len(data) + 7) // 8
        byte_data = int(data, 2).to_bytes(byte_len, byteorder='big')
        with open(filename, 'wb') as f:
            f.write(byte_data)
        # save the exact bit length in a sidecar .len file
        with open(filename + ".len", 'w') as lf:
            lf.write(str(len(data)))
    else:
        with open(filename, 'w') as f:
            f.write(data)
    print(f"Saved file: {filename}, Size: {os.path.getsize(filename)} bytes")

def read_from_file(filename, binary=False):
    """
    If binary=True expects a companion .len file with exact bit length, otherwise uses file-size*8 (may include padding).
    If binary=False reads exact text bitstring (no padding).
    """
    if binary:
        with open(filename, 'rb') as f:
            byte_data = f.read()
        # get exact bit length if available
        len_file = filename + ".len"
        if os.path.exists(len_file):
            with open(len_file, 'r') as lf:
                bit_len = int(lf.read().strip())
        else:
            bit_len = len(byte_data) * 8
        bit_string = bin(int.from_bytes(byte_data, byteorder='big'))[2:]
        # pad leading zeros to match byte length
        if len(bit_string) < len(byte_data) * 8:
            bit_string = '0' * (len(byte_data) * 8 - len(bit_string)) + bit_string
        # truncate to the original bit length (strip padding)
        if len(bit_string) > bit_len:
            bit_string = bit_string[-bit_len:]
        return bit_string
    else:
        with open(filename, 'r') as f:
            return f.read().strip()

# Function to calculate frequencies of elements in data
def calculate_frequencies(data):
    freq = {}
    for value in data:
        freq[value] = freq.get(value, 0) + 1
    return freq

# Function to build Huffman tree
def build_huffman_tree(freq_dict):
    priority_queue = [HuffmanNode(symbol, freq) for symbol, freq in freq_dict.items()]
    heapq.heapify(priority_queue)
    while len(priority_queue) > 1:
        left, right = heapq.heappop(priority_queue), heapq.heappop(priority_queue)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left, merged.right = left, right
        heapq.heappush(priority_queue, merged)
    return priority_queue[0]

# Function to generate Huffman codes
def generate_huffman_codes(node, prefix="", code_table=None):
    if code_table is None:
        code_table = {}
    if node:
        if node.symbol is not None:
            code_table[node.symbol] = prefix
        generate_huffman_codes(node.left, prefix + "0", code_table)
        generate_huffman_codes(node.right, prefix + "1", code_table)
    return code_table

# Function to encode data using Huffman codes
def huffman_encode(data, code_table):
    return "".join(code_table[value] for value in data)

# Function to decode data using Huffman codes
def huffman_decode(encoded_data, code_table):
    # Create reverse mapping
    reverse_code_table = {v: k for k, v in code_table.items()}
    
    decoded_data = []
    current_code = ""
    
    # Ensure encoded_data is a string
    if not isinstance(encoded_data, str):
        encoded_data = str(encoded_data)
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_code_table:
            decoded_data.append(reverse_code_table[current_code])
            current_code = ""
    
    return decoded_data

def iterative_huffman_compression(data, iterations=1):
    """
    Iteratively apply Huffman but save the exact bitstring to .txt each iteration.
    NOTE: Iterating Huffman on the already-encoded bitstring is usually not necessary.
    """
    code_tables = []
    current_data = data

    for i in range(1, iterations + 1):
        original_length = len(current_data)
        freq_dict = calculate_frequencies(current_data)
        # Need at least two symbols for Huffman; if only 1 symbol, create trivial table
        if len(freq_dict) == 1:
            # create mapping of the single symbol -> '0'
            single_symbol = next(iter(freq_dict.keys()))
            huffman_code_table = {single_symbol: "0"}
            huffman_tree = None
        else:
            huffman_tree = build_huffman_tree(freq_dict)
            huffman_code_table = generate_huffman_codes(huffman_tree)
        code_tables.append({str(k): v for k, v in huffman_code_table.items()})  # stringify keys for JSON

        # Encode (current_data may be list of ints or a bitstring). Ensure we treat symbols properly:
        # If current_data is a list, convert elements to their original symbol type (string) for consistent mapping
        if isinstance(current_data, list):
            # map each element to its code (convert element to same type used in the code_table keys)
            # code_tables stored keys as strings — convert symbol to string when looking up
            encoded = "".join(huffman_code_table.get(int(x) if isinstance(x, (int, np.integer)) else x, 
                                                   huffman_code_table.get(str(x), "")) for x in current_data)
        else:
            # if it's already a string of symbols (like '0'/'1' characters), treat each character as symbol
            encoded = "".join(huffman_code_table.get(ch, "") for ch in current_data)
        current_data = encoded
        compressed_length = len(current_data)
        # Save exact bitstring to text (reliable) and also binary + length (optional)
        save_to_file(current_data, f"compressed_{i}.txt", binary=False)
        save_to_file(current_data, f"compressed_{i}.bin", binary=True)
        print(f"Iteration {i}: Original size = {original_length} symbols, Compressed size = {compressed_length} bits")

    # Save code tables for decompression (they are already string-keyed)
    with open("code_tables.json", "w") as f:
        json.dump(code_tables, f)

    return current_data, code_tables

def iterative_huffman_decompression(compressed_bitstring, code_tables):
    """
    Decompress by applying code tables in reverse order. Expects compressed_bitstring to be exact bitstring (no padding).
    code_tables is a list of dicts where keys are stringified symbols.
    """
    current_data = compressed_bitstring
    for i in range(len(code_tables)-1, -1, -1):
        table = code_tables[i]
        # reverse mapping: code -> symbol (symbol kept as string)
        reverse_code_table = {v: k for k, v in table.items()}

        decoded = []
        cur = ""
        for bit in current_data:
            cur += bit
            if cur in reverse_code_table:
                decoded.append(reverse_code_table[cur])
                cur = ""
        if cur != "":
            # leftover bits -> indicates mismatch or padding
            # we'll keep them ignored or raise warning
            print(f"Warning: leftover bits after iteration {i+1}: '{cur}' (likely padding or mismatch)")

        # After decoding, decoded is a list of symbol-strings. For next iteration we need a *string* of symbols
        current_data = "".join(str(s) for s in decoded)

        print(f"Decompression iteration {i+1}: Decoded size = {len(decoded)} symbols")

    # final output returned as list of ints if symbols are numeric strings
    final = [int(s) if s.isdigit() else s for s in decoded]
    return final

# Function to reconstruct QR from shares
def reconstruct_qr(share1, share2):
    return np.bitwise_xor(share1, share2)

# Main function
if __name__ == "__main__":
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")
    share1, share2 = create_shares(qr_array)
 ### ADDED: Original QR entropy analysis
    qr_binary = "".join("1" if bit else "0" for row in qr_array for bit in row)
    qr_zeros = qr_binary.count("0")
    qr_ones = qr_binary.count("1")
    qr_entropy = shannon_entropy(qr_binary)
    print("\n=== Original QR Code Binary ===")
    print(f"Size: {len(qr_binary)} bits")
    print(f"Number of 0s: {qr_zeros}, Number of 1s: {qr_ones}")
    print(f"Shannon Entropy: {qr_entropy:.4f} bits/symbol\n")
    
    # Convert share1 to a list of integers
    vector1 = [int(x) for x in share1.ravel()]
    original_size = len(vector1)
    print(f"Original share size: {original_size} elements")

     # ---- Entropy after VC ----
    entropy_share1 = shannon_entropy(share1)
    num_zeros_s1 = np.count_nonzero(share1 == 0)
    num_ones_s1 = np.count_nonzero(share1 == 1)
    print("\n--- Share1 Statistics ---")
    print(f"Share1 size: {share1.size} bits")
    print(f"Number of 0s: {num_zeros_s1}, Number of 1s: {num_ones_s1}")
    print(f"Shannon Entropy (Share1): {entropy_share1:.4f} bits/symbol")


    
    # Apply Huffman compression 10 times
    compressed_data, code_tables = iterative_huffman_compression(vector1, 10)
    
    # Save the final compressed data for fallback
    save_to_file(compressed_data, "final_compressed.bin", binary=True)
    save_to_file(compressed_data, "final_compressed.txt")
    
    # Comment out QR embedding (optional)
    '''
    # Embed the final compressed data into a QR code
    generate_qr(compressed_data, "compressed_qr.png")
    print("Compressed data embedded in QR code: compressed_qr.png")
    '''
    
    # DECOMPRESSION SECTION
    print("\n--- Starting Decompression ---")
    
    # Load code tables
    with open("code_tables.json", "r") as f:
        code_tables_loaded = json.load(f)
        # Convert back to original format
        code_tables = [{int(k): v for k, v in table.items()} for table in code_tables_loaded]
    
    # Get compressed data (from QR or from file)
    # Comment out QR extraction (optional)
    '''
    # In a real implementation, you would extract data from the QR code
    # For this example, we'll read from the saved file
    compressed_data_from_qr = read_from_file("final_compressed.bin", binary=True)
    '''
    # Use fallback data if QR extraction is commented
    compressed_data_from_file = read_from_file("final_compressed.bin", binary=True)
    # Load code tables (they were saved as string-keyed dicts)
    with open("code_tables.json", "r") as f:
        code_tables_loaded = json.load(f)  # list of dicts with string keys and bitstring values

    # Read compressed bitstring reliably from compressed_N.txt (use text file to preserve exact bits)
    compressed_bitstring = read_from_file("compressed_10.txt", binary=False)  # replace 10 with your last iter number

    # Decompress
    decompressed_list = iterative_huffman_decompression(compressed_bitstring, code_tables_loaded)
    
    # Apply iterative decompression
    #decompressed_data = iterative_huffman_decompression(compressed_data_from_file, code_tables)
    
    # Check if the decompressed data has the correct length
    if len(decompressed_list) != original_size:
        print(f"Warning: Decompressed size ({len(decompressed_list)}) doesn't match expected size ({original_size})")
        # Truncate or pad to match expected size
        if len(decompressed_list) > original_size:
            decompressed_data = decompressed_list[:original_size]
        else:
            decompressed_list.extend([0] * (original_size - len(decompressed_list)))
    
    # Reshape to original share shape
    decompressed_share = np.array(decompressed_list, dtype=np.uint8).reshape(share1.shape)
    
    # Reconstruct QR code from shares
    reconstructed_qr = reconstruct_qr(decompressed_share, share2)
    
    # Save reconstructed QR code
    Image.fromarray(reconstructed_qr * 255).save("reconstructed_qr.png")
    
    # Verify the decompression
    if np.array_equal(share1, decompressed_share):
        print("SUCCESS: Decompressed share matches original exactly!")
        print(f"Original size: {len(vector1)} elements")
        print(f"Decompressed size: {len(decompressed_list)} elements")
    else:
        print("ERROR: Decompressed share does not match original!")
        # Count differences
        differences = np.sum(share1 != decompressed_share)
        print(f"Number of differing elements: {differences}")
    
    # Verify QR reconstruction
    if np.array_equal(qr_array, reconstructed_qr):
        print("The QR reconstruction was successful! The QR code matches the original.")
    else:
        print("QR reconstruction failed: The QR code does not match the original.")