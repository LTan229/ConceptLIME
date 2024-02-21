import itertools
import math
import os
from urllib.parse import urlparse
import pickle
import numpy as np
import matplotlib.colors as mcolors


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


def mask_imgs(zs: np.ndarray, 
                segmentation: np.ndarray, 
                image: np.ndarray,
                color="average") -> np.ndarray:
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                if color == "average":
                    out[i][segmentation == j,:] = np.sum(image[segmentation == j], axis=(0)) / np.sum(segmentation == j)
                else:
                    out[i][segmentation == j,:] = np.array(mcolors.to_rgb(color)) * 255
    return out


def generate_combinations(comb_len, remove_cnt, comb_num):
    if math.comb(comb_len, remove_cnt) < 3 * comb_num:
        combinations = []
        for ids in itertools.combinations(range(comb_len), remove_cnt):
            combination = np.ones(comb_len)
            combination[list(ids)] = 0
            combinations.append(combination)
        combinations = np.array(combinations)
        if len(combinations) > comb_num:
            ids = np.random.choice(len(combinations), comb_num, replace=False)
            combinations = combinations[ids]
        return combinations
    else:
        combinations = []
        while len(combinations) < comb_num:
            ids = np.random.choice(comb_len, remove_cnt, replace=False)
            combination = np.ones(comb_len)
            combination[ids] = 0
            combination = combination.tolist()
            if combination not in combinations:
                combinations.append(combination)
        return np.array(combinations)