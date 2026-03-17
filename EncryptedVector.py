import numpy as np

class EncryptedVector:
    """
    Simulates CKKS ciphertext.
    """

    def __init__(self, data):
        self.data = np.array(data)

    def add(self, other):
        return EncryptedVector(self.data + other.data)

    def multiply(self, other):
        return EncryptedVector(self.data * other.data)

    def rotate(self, steps):
        return EncryptedVector(np.roll(self.data, steps))

    def sum(self):
        return np.sum(self.data)

    def decrypt(self):
        return self.data
