import numpy as np
import torch
import os
from test import test
from utils import random_anchor_and_positive_within_class, random_negative_of_anchor, \
    distance_between_vectors, gen_samples_for_selection, read_images_to_tensor


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, strategy='random', model=None, test_path=None, neg_size=20):
        self.data = data_path
        self.test_path = test_path[0]
        self.test_dataset_path = test_path[1]
        self.person_num_photos = {}
        self._get_pairs()
        self.strategy = strategy
        self.transform = transform
        self.model = model
        self.neg_size = neg_size
        self.k = 5
        self.dist_func = lambda x, y: torch.cdist(x, y)

        if self.strategy in ('hard-negative', 'semi-hard-negative'):
            assert self.model is not None
            assert self.test_path is not None

    def _get_pairs(self):
        '''
         Creates dict of all names
         Creates dict of names of people with more than one photo
        '''
        for folder in os.listdir(self.data_path):
            self.person_num_photos[folder] = len(os.listdir(os.path.join(self.data_path, folder)))
        self.pairs = {}
        self.id_to_pair = {}
        index = 0
        for folder in os.listdir(self.data_path):
            if self.person_num_photos[folder] >= 2:
                self.pairs[folder] = self.person_num_photos[folder]
                self.id_to_pair[index] = folder
                index += 1

    def update_model(self, model):
        '''
         Update model weights for feature extraction
        :param model: model
        '''
        self.model.load_state_dict(model.state_dict())

    def update_negative_sample_size(self):
        '''
         Calculates the size of the negative elements.
         Calculated by the probabilistic number of negative items that are
          less than the first quartile of distances to positive items
        '''
        distances = test(self.model, self.test_path, self.test_dataset_path, self.transform, self.dist_func)
        pos_quartile = np.nanquantile(np.where(distances[:, 1] == 0, distances[:, 0], np.nan), 0.25)
        negatives_mask = np.where((distances[:, 1] == 1) & (distances[:, 1] < pos_quartile), distances[:, 0], np.nan)
        neg_size = np.count_nonzero(~np.isnan(negatives_mask))
        neg_all_size = np.count_nonzero(~np.isnan(np.where(distances[:, 1] == 0, distances[:, 0], np.nan)))
        self.neg_size = self.k * neg_all_size // neg_size

    def random_mining(self, person):
        '''
         Random strategy for triplet generation
        :param person: anchor person
        :return: tensor of anchor, positive and negative images
        '''
        anchor, positive = random_anchor_and_positive_within_class(self.pairs, person)
        negative = random_negative_of_anchor(self.personid_to_dataidx, person)
        anchor = read_images_to_tensor(self.data, anchor)
        positive = read_images_to_tensor(self.data, positive)
        negative = read_images_to_tensor(self.data, negative)

        return anchor, positive, negative

    def semi_hard_negative_mining(self, person, margin=0.75):
        '''
         Semi-hard strategy for triplet generation
        :param person: anchor person
        :return: tensor of anchor, positive and negative images
        '''
        anchor, positives, negatives = gen_samples_for_selection(self.pairs, self.photos, person, self.neg_size)

        anchor = read_images_to_tensor(self.data, anchor).to(torch.float32)
        positive = read_images_to_tensor(self.data, positives).to(torch.float32)
        negative = read_images_to_tensor(self.data, negatives).to(torch.float32)

        dist_positive = distance_between_vectors(self.model, anchor, positive)
        dist_negative = distance_between_vectors(self.model, anchor, negative)

        index_distant_positive = dist_positive.argmax().item()
        distant_positive = positive[index_distant_positive, ...]
        distance_to_positive = dist_positive[0, index_distant_positive]

        max_negative_dist = dist_negative.max()
        semi_negative_mask = torch.where(dist_negative > (distance_to_positive + margin), dist_negative,
                                         max_negative_dist)
        semi_hard_negative = negatives[semi_negative_mask.argmin().item(), ...]

        return anchor, distant_positive, semi_hard_negative

    def hard_negative_mining(self, person):
        '''
         Hard strategy for triplet generation
        :param person: anchor person
        :return: tensor of anchor, positive and negative images
        '''
        anchor, positives, negatives = gen_samples_for_selection(self.pairs, self.photos, person, self.neg_size)

        anchor = read_images_to_tensor(self.data, anchor).to(torch.float32)
        positive = read_images_to_tensor(self.data, positives).to(torch.float32)
        negative = read_images_to_tensor(self.data, negatives).to(torch.float32)

        dist_positive = distance_between_vectors(self.model, anchor, positive)
        dist_negative = distance_between_vectors(self.model, anchor, negative)

        distant_positive = positives[dist_positive.argmax().item(), ...]
        hard_negative = negatives[dist_negative.argmin().item(), ...]

        return anchor, distant_positive, hard_negative

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        person = self.id_to_pair[idx]
        if self.strategy == 'random':
            anchor, positive, negative = self.random_mining(person)
        elif self.strategy == 'semi-hard-negative':
            anchor, positive, negative = self.semi_hard_negative_mining(person, margin=0.0)
        elif self.strategy == 'hard-negative':
            anchor, positive, negative = self.hard_negative_mining(person)
        else:
            raise ValueError

        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        return anchor, positive, negative
