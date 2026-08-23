<<<<<<< Updated upstream
=======
import math

def encode(plain_text, a, b):
    m = 26
    if math.gcd(a, m) != 1:
        raise ValueError("a and m must be coprime")
    letters = [c for c in plain_text if c.isalpha()]
    x_values = [ord(c) - ord('a') for c in letters]
    encrypted = [(a * x + b) % m for x in x_values]
    ciphertext = ''.join([chr(x + ord('a')) for x in encrypted])
    return ciphertext

def decode(ciphered_text, a, b):
    m = 26
    if math.gcd(a, m) != 1:
        raise ValueError("a and m must be coprime")
    letters = [c for c in ciphered_text if c.isalpha()]
    if not letters:
        return ''
    y_values = [ord(c) - ord('a') for c in letters]
    a_inv = pow(a, -1, m)
    decrypted = [(a_inv * (y - b)) % m for y in y_values]
    plaintext = ''.join([chr(x + ord('a')) for x in decrypted])
    return plaintext
>>>>>>> Stashed changes
