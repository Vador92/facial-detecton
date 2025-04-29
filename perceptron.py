import random
import time
import os

from typing import Tuple, List

# Data Loading Helper

def load_data_from_txt(filepath: str, image_size: Tuple[int, int], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load digit or face data from the .txt 
    Each image stored as 'image_size' lines of text, followed by label
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    images, labels = [], []
    i = 0
    while i < len(lines):
        image = []
        for _ in range(image_size[0]):
            row = [1 if ch != ' ' else 0 for ch in lines[i].strip('\n')]
            image.extend(row)
            i += 1
        label = int(lines[i].strip())
        i += 1
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)