import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image
import os


def distance_between_vectors(model, tensor_of_images_1: torch.tensor, tensor_of_images_2: torch.tensor):
    '''
     Calculates distance between features of two images
    :param model: model to compute image features
    :param tensor_of_images_1: first image
    :param tensor_of_images_2: second image
    :return: distance
    '''
    with torch.no_grad():
        feature_v1 = model(tensor_of_images_1)
        feature_v2 = model(tensor_of_images_2)
    if len(feature_v1.shape) < 2:
        feature_v1 = feature_v1.unsqueeze(0)
    if len(feature_v2.shape) < 2:
        feature_v2 = feature_v2.unsqueeze(0)
    if len(feature_v1.shape) != 2 and len(feature_v2.shape) != 2:
        raise ValueError
    return torch.cdist(feature_v1, feature_v2)


def random_anchor_and_positive_within_class(pairs: dict, person: str, size: int = 2):
    '''
     Returns anchor and positive elements
    :param pairs: dict of people who have more than three photos
    :param person: anchor person
    :param size: size of positive photos to use
    :return: array (anchor person, position in folder) and array (positive person, position in folder)
    '''
    positive_idx = np.arange((0, pairs.get(person)))
    if size == -1:
        size = len(positive_idx)
    positives = np.random.choice(positive_idx, size=size, replace=False)
    pos_anchor, pos_positives = positives[0], positives[1:]
    return [(person, pos_anchor)], [(person, pos_p) for pos_p in pos_positives]


def random_negative_of_anchor(photos: dict, anchor: str, size: int = 1):
    '''
     Returns negative elements of anchor
    :param photos: dict of all people
    :param anchor: anchor person
    :param size: number of elements to return
    '''
    negatives = []
    while len(negatives) < size:
        negative_el = np.random.choice(list(photos.keys()), size=1).item()
        if negative_el != anchor:
            negatives.append((negative_el, np.random.choice(np.arange((0, photos.get(negative_el))), size=1).item()))
    return negatives


def gen_samples_for_selection(pairs: dict, photos: dict, person: str, neg_size: int):
    '''
     Generates data for triplet selection
    :param pairs: dict of people who have more than three photos
    :param photos: dict of all people
    :param person: anchor person
    :param neg_size: number of negative elements to generate
    :return: anchor, positives, negatives
    '''
    shuffled_positives = random_anchor_and_positive_within_class(pairs, person, size=-1)
    anchor, positives = shuffled_positives[0], shuffled_positives[1:]
    negatives = random_negative_of_anchor(photos, person, neg_size)
    return anchor, positives, negatives


def read_images_to_tensor(path_to_dataset: str, person_indexes: [(str, int)]):
    '''
     Images to tensor
    :param path_to_dataset: dataset path
    :param person_indexes: person name
    :return: tensor of photos
    '''
    transform = v2.Compose([v2.PILToTensor()])
    tensor = torch.full((len(person_indexes), 3, 96, 96), 0.0)
    for index, (person, number) in enumerate(person_indexes):
        path_to_folder = os.path.join(path_to_dataset, person)
        image = Image.open(f'{path_to_folder}/image_{number}.jpg')
        tensor[index, ...] = transform(image)
    return tensor