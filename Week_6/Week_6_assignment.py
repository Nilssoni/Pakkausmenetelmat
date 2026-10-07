# Huffman coding algorithm implementation in Python

class Node:
    def __init__(self, byte=None, freq=None):
        self.byte = byte
        self.freq = freq
        self.left = None
        self.right = None

nodes = []

def calculate_frequencies(data):
    frequencies = {}

    for byte in data:
        if byte in frequencies:
            frequencies[byte] += 1
        else:
            frequencies[byte] = 1
    return frequencies

def build_huffman_tree(frequencies):
    nodes = []
    for byte, freq in frequencies.items():
        nodes.append(Node(byte, freq))
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged = Node(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right
        nodes.append(merged)
    return nodes[0]

def generate_huffman_codes(node, current_code="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return
    if node.byte is not None:
        codes[node.byte] = current_code
        return codes
    generate_huffman_codes(node.left, current_code + "0", codes)
    generate_huffman_codes(node.right, current_code + "1", codes)
    return codes

def huffman_encoding(data):
    frequencies = calculate_frequencies(data)
    root = build_huffman_tree(frequencies)
    huffman_codes = generate_huffman_codes(root)
    return frequencies, huffman_codes

def __main__():
    with open(r"C:\Codes\Pakkausmenetelmat\Week_6\alice_in_wonderland.txt", "rb") as file:
        data = file.read()
    frequencies = calculate_frequencies(data)
    root = build_huffman_tree(frequencies)
    codes = generate_huffman_codes(root)
    print("Byte\tOccurrences\tHuffman Code")
    for byte, freq in sorted(
        frequencies.items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        print(f"0x{byte:02X}\t{freq}\t\t{codes[byte]}")

if __name__ == "__main__":
    __main__()