import os
import numpy as np
import torch
from data.utils import read_images_to_tensor


def get_indexes(index1, index2, bound):
    '''
     If the indexes exceed the number of images in the folder,
      then the indexes are updated randomly
    :param index1: index of the first image
    :param index2: index of the second image
    :param bound: number of images in folder
    :return: indexes
    '''
    if index1 >= bound or index2 >= bound:
        index1, index2 = np.random.choice(np.arange(0, bound), replace=False, size=2)
    return index1, index2


def test(model, test_file_path, test_dataset_path, transform, dist_func):
    '''
     Function to calculate distances with tags
    :param model: model
    :param test_file_path: path to a file with names and image indexes
    :param test_dataset_path: dataset path
    :param transform: transformation for tensor normalization
    :param dist_func: function to calculate distance between two feature vectors
    '''
    model.eval()
    distances = []
    with open(test_file_path) as f:
        test_pairs = f.read().split('\n')
    for pair in test_pairs:
        pair = pair.split('\t')

        if len(pair) == 3:
            name, index1, index2 = pair
            index1, index2 = int(index1), int(index2)
            is_diff = False
            bound = len(os.listdir(os.path.join(test_dataset_path, name)))
            index1, index2 = get_indexes(index1 - 1, 0, bound)
            imgs_tensor = read_images_to_tensor(test_dataset_path[0], [(name, index1), (name, index2)])
            img1_tensor = imgs_tensor[0, ...]
            img2_tensor = imgs_tensor[1, ...]
        else:
            name1, index1, name2, index2 = pair
            is_diff = True

            index1, index2 = int(index1), int(index2)
            bound1 = len(os.listdir(os.path.join(test_dataset_path, name1)))
            index1, _ = get_indexes(index1 - 1, 0, bound1)

            bound2 = len(os.listdir(os.path.join(test_dataset_path, name2)))
            index2, _ = get_indexes(index2 - 1, 0, bound2)

            img1_tensor = read_images_to_tensor(os.path.join(test_dataset_path, name1), [(name1, index1)])
            img2_tensor = read_images_to_tensor(os.path.join(test_dataset_path, name2), [(name2, index2)])

        with torch.no_grad():
            v1 = model(transform(img1_tensor).unsqueeze(0))
            v2 = model(transform(img2_tensor).unsqueeze(0))
        distances.append([dist_func(v1, v2).item(), is_diff])
    return np.array(distances)
