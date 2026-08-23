import math

def encode(plain_text, a, b):
    if not plain_text:
        return ""
    m = 26
    if math.gcd(a, m) != 1:
        raise ValueError("a and m must be coprime.")
    numeric_chars = []
    for c in plain_text:
        if c.isalpha():
            x = ord(c.lower()) - ord('a')
            numeric_chars.append(x)
    encoded_chars = []
    for x in numeric_chars:
        y = (a * x + b) % m
        encoded_char = chr(y + ord('a'))
        encoded_chars.append(encoded_char)
    return ''.join(encoded_chars)
    
def decode(ciphertext, a, b):
    if not ciphertext:
        return ""
    m = 26
    if math.gcd(a, m) != 1:
        raise ValueError("a and m must be coprime.")
    a_inv = pow(a, -1, m)
    decoded_chars = []
    for c in ciphertext:
        if c.isalpha():
            y = ord(c.lower()) - ord('a')
            x = (a_inv * (y - b)) % m
            decoded_char = chr(x + ord('a'))
            decoded_chars.append(decoded_char)
    return ''.join(decoded_chars)