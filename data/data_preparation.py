import tarfile
import os
import dlib
import cv2
import torch
from torchvision.transforms import v2
from tqdm import tqdm
#import json

MINPP = 3   # MIN IMAGES NUMBER PER PERSON
MAXPP = 20  # MAX IMAGES NUMBER PER PERSON


def crop_face(image, face_detector, res_shape=(96, 96)):
    '''
    :param image: numpy array. image of a person
    :param face_detector: dlib face detector
    :param res_shape: resize detected face to res_shape
    :return: numpy array or None if face is not detected
    '''
    try:
        rect = face_detector(image, upsample_num_times=3)[0]
        start_x = rect.left()
        start_y = rect.top()
        end_x = rect.right()
        end_y = rect.bottom()
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(end_x, image.shape[1])
        end_y = min(end_y, image.shape[0])
        image = image[start_y:end_y, start_x:end_x]
        return cv2.resize(image, res_shape)
    except TypeError or IndexError:
        return None


def find_size_of_tensor(path_to_dataset):
    '''
     Finds the number of all dataset given the minimum and maximum number of photos for a class
    :param path_to_dataset: path to dataset
    :return: final number of photos
    '''
    global MINPP, MAXPP
    dataset_size = 0
    for person in os.listdir(path_to_dataset):
        path_to_folder = os.path.join(path_to_dataset, person)
        length = len(os.listdir(path_to_folder))
        if length == 0:
            continue
        dataset_size += min(max(MINPP, length), MAXPP) if length > 1 else 1
    return dataset_size


def read_images_from_folder(path_to_folder, face_detector, length):
    '''
     Reads photos from a folder and converts them to a tensors of photos and detected faces
    :param path_to_folder: dataset folder path
    :param face_detector: dlib face detector
    :param length: number of photos for detection
    :return: tensor of photos, tensor of faces, number of detected faces
    '''
    person_photos = torch.zeros((length, 3, 250, 250))
    person_faces = torch.zeros((length, 3, 96, 96))
    detect_faces = torch.full((length, ), False)
    face_idx = 0
    for file in os.listdir(path_to_folder):
        image = cv2.imread(os.path.join(path_to_folder, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (250, 250))
        detected_face = crop_face(image, face_detector)
        if detected_face is not None:
            person_photos[face_idx, ...] = torch.from_numpy(image).permute(2, 0, 1)
            person_faces[face_idx, ...] = torch.from_numpy(detected_face).permute(2, 0, 1)
            detect_faces[face_idx] = True
            face_idx += 1
    return person_photos, person_faces[detect_faces], face_idx


def write_to(tensor, path_to_write):
    '''
     Writes photos from the tensor into a folder
    :param tensor: tensor of photos
    :param path_to_write: dataset folder path
    '''
    for image_id, image in enumerate(tensor):
        image = image.permute(1, 2, 0)
        cv2.imwrite(f'{path_to_write}/image_{image_id}.jpg', image)


def add_augment(person: torch.tensor, face_detector, transformation, face_idx):
    '''
     Add augmentation to a photos from read_images_from_folder function. Then cut faces off and writes to a tensor.
    :param person:
    :param face_detector:
    :param transformation:
    :param face_idx:
    :return:
    '''
    aug_person_faces = torch.zeros((face_idx, 3, 96, 96))
    detect_faces = torch.full((face_idx, ), False)
    for idx in range(face_idx):
        image = transformation(person[idx]).permute(1, 2, 0).numpy().astype('uint8')
        detected_face = crop_face(image, face_detector)
        if detected_face is not None:
            aug_person_faces[idx, ...] = torch.from_numpy(detected_face).permute(2, 0, 1)
            detect_faces[idx] = True
    aug_person_faces = aug_person_faces[detect_faces]
    return aug_person_faces


def augment_dataset(path_to_dataset, face_detector, transformation, save_to_tensor=False, tensor_size=None):
    '''
     Reads and adds augmentation to a dataset images
    :param path_to_dataset: dataset path
    :param face_detector: dlib face detector
    :param transformation: transformation for augmentation
    :param save_to_tensor: flag to save face images into a tensor
    :param tensor_size: tensor size
    :return:
    '''
    global MINPP, MAXPP

    person_id = 0
    person_faces_index = 0
    personid_to_idx = {}
    name_to_personid = {}

    if save_to_tensor:
        faces = torch.full((tensor_size, 3, 96, 96), 0).to(torch.uint8)

    for folder in tqdm(os.listdir(path_to_dataset)):
        path_to_folder = os.path.join(path_to_dataset, folder)
        length = len(os.listdir(path_to_folder))

        if length == 0:
            continue
        final_length = min(max(MINPP, length), MAXPP) if length > 1 else 1

        person_photos, person_faces, face_idx = read_images_from_folder(path_to_folder, face_detector, length)

        if face_idx == 0:
            continue
        if face_idx < final_length:
            aug_person_faces, num_detected_faces = add_augment(person_photos, face_detector, transformation, face_idx)
            if num_detected_faces > 0:
                person_faces = torch.vstack((person_faces, aug_person_faces))

        num_faces = len(person_faces)
        if save_to_tensor:
            personid_to_idx[person_id] = (person_faces_index, person_faces_index + num_faces)
            name_to_personid[folder] = person_id
            faces[person_faces_index:person_faces_index + num_faces] = person_faces.to(torch.uint8)
            person_faces_index += num_faces
        else:
            write_to(person_faces, path_to_folder)
        person_id += 1

    if save_to_tensor:
        return faces, person_faces_index, personid_to_idx


if __name__ == '__main__':
    transform = v2.Compose([v2.RandomRotation(degrees=(-15, 15)),
                            v2.RandomHorizontalFlip(p=0.5),
                            v2.RandomPerspective(distortion_scale=0.2, p=1.0),
                           ])

    path_to_dataset = '...'

    files = tarfile.open(os.path.join(path_to_dataset, 'lfw.tar'))
    files.extractall(path_to_dataset)

    tensor_size = find_size_of_tensor(path_to_dataset)
    print(tensor_size)
    faces = torch.zeros((tensor_size, 3, 96, 96)).to(torch.uint8)
    detector = dlib.get_frontal_face_detector()

    augment_dataset(path_to_dataset, detector, transform)