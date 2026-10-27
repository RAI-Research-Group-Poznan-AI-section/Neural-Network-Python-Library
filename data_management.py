import os
import numpy as np
from PIL import Image


def load_labels(labels_path: str, num_of_numbers: int = 10) -> list[int]:
    """
    Reads labels from a file, in order.

    Supported formats:
    .npy: a numpy array.
    any text file: the format of labels either has to be a single column of labels or a header row followed by a single column of labels.

    Parameters:
    labels_path (str): The labels file.
    num_of_numbers (int): The number of different labels (e.g. 10 for digits 0-9).

    Returns:
    list[int]: The labels as ints.

    Raises:
    ValueError: If a label is not a digit 0 to 9.
    """
    if os.path.splitext(labels_path)[1].lower() == ".npy":
        raw_labels = [str(v) for v in np.load(labels_path).ravel()]
    else:
        with open(labels_path, encoding="utf-8-sig") as f:
            raw_labels = f.read().split()
        if raw_labels and raw_labels[0].isalpha():
            raw_labels = raw_labels[1:]

    labels = []
    for i, value in enumerate(raw_labels):
        if value not in [str(d) for d in range(num_of_numbers)]:
            raise ValueError(f"Label {i} in '{labels_path}' must be a digit 0 to {num_of_numbers - 1}, got '{value}'")
        labels.append(int(value))
    return labels


def pictures_load(picture_path: str) -> list[np.ndarray]:
    """
    Loads every grayscale picture in a folder as a column vector, in file name order.

    Parameters:
    picture_path (str): Folder with the pictures.

    Returns:
    list[np.ndarray]: The pictures as numpy arrays of shape (width * height, 1) with values in [0, 1] for sigmoid activation.

    Raises:
    ValueError: If the path is not a folder, it has no pictures.
    For now it only supports grayscale images as there is no need for color images in this project.
    """
    if not os.path.isdir(picture_path):
        raise ValueError(f"'{picture_path}' is not a folder")

    pictures = []
    for filename in sorted(os.listdir(picture_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            img = Image.open(os.path.join(picture_path, filename))
            if img.mode != "L":
                # for now it only supports grayscale images as there is no need for color images in this project
                raise ValueError(f"Picture '{filename}' must be grayscale, got mode '{img.mode}'") 
            pictures.append(np.asarray(img, dtype=np.float64).reshape(-1, 1) / 255)  # normalize to [0, 1]

    if not pictures:
        raise ValueError(f"No pictures found in '{picture_path}'")
    return pictures


