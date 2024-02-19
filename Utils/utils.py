import os
from urllib.parse import urlparse
import pickle
import numpy as np


def download_file(url: str, directory: str):
    file_name = os.path.basename(urlparse(url).path)
    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        os.system(f'wget -P {directory} {url}')


def ensure_path(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_pickle(file_name: str, obj, mode="wb"):
    with open(file_name, mode) as f:
        pickle.dump(obj, f)


def fill_segmentation(values, segmentation):
    """it is desired that segmentation starts from 0, otherwise the head of values should be padded"""
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out