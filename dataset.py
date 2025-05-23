from torchvision import transforms
from torchvision.datasets import VisionDataset
import pickle, os
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from typing import List
import skimage.filters

data_array_file = 'https://www.cv.nrao.edu/~bkent/astro/vlass/vlass_data_array.p'
labels_file = 'https://www.cv.nrao.edu/~bkent/astro/vlass/vlass_labels.p'

class ImageDataset(VisionDataset):
    # Label mapping
    # { 
    #     0: 'Extended',
    #     1: 'Point Source',
    #     2: 'Double Lobes',
    #     3: 'Three Points'
    # }
    
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    
class VLASS(VisionDataset):
    def __init__(
        self, root = None, split: str = None, seed=42,
        test_size=0.2,
        transforms = None, 
        transform = None, target_transform = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        x, y = get_data(data_folder=root)
        
        if split is None:
            self.x = x
            self.y = y
            return
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, 
            shuffle=True, stratify=y
        )
        if split == 'train':
            self.x = x_train
            self.y = y_train
        elif split == 'test':
            self.x = x_test
            self.y = y_test
        else:
            raise ValueError("split must be 'train' or 'test'")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.x[idx], dtype=torch.float32) # self.x[idx] # 
        target = self.y[idx]
        
        if self.transform is not None:
            image, target = self.transforms(image, target)
        
        return np.array(image), target
    
def get_data(data_folder='./data/'):
    """
    Returns the VLASS data and labels. 
    Downloads the data if necessary.
    """
    
    url_root = 'https://www.cv.nrao.edu/~bkent/astro/vlass/'
    
    data_array_file = 'vlass_data_array.p'
    labels_file = 'vlass_labels.p'
    
    file_path = os.path.join(data_folder, data_array_file)
    if not os.path.exists(file_path):
        data_array_file = url_root + data_array_file
        print("Downloading data from URL: {!s}".format(data_array_file))
        data_array = pickle.load(urlopen(data_array_file))
    else:
        print("Loading data from file: {!s}".format(file_path))
        data_array = pickle.load(open(file_path, 'rb'))
       
    file_path = os.path.join(data_folder, labels_file) 
    if not os.path.exists(file_path):
        labels_file = url_root + labels_file
        print("Downloading labels from URL: {!s}".format(labels_file))
        labels = pickle.load(urlopen(labels_file))
    else:
        print("Loading labels from file: {!s}".format(file_path))
        labels = pickle.load(open(file_path, 'rb'))
        
    da = data_array.shape
    dl = labels.shape
    print("{!s} images, each of size {!s} x {!s} pixels.".format(da[0],da[1],da[2]))
    print("There are {!s} corresponding labels - one category for each image.".format(dl[0]))
    
    data_array = data_array.reshape(-1, 1, 64, 64)

    return data_array, labels

class DataAugmentations:
    def __init__(
        self, image_size,
        hflip_prob=0.2, 
        vflip_prob=0.2,
        blur_prob=0.1,
        noise_prob=0.1,
        normalize=True
    ):
        transformations = [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=hflip_prob),
            transforms.RandomVerticalFlip(p=vflip_prob),
            RandomGaussianBlur(p=blur_prob),
            RandomGaussianNoise(im_dim=image_size, p=noise_prob)
        ]
        if normalize:
            # Normalize the image
            transformations.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        self.transformations = transforms.Compose(transformations)
        
    def __call__(self, image):
        return self.transformations(image)


class RandomGaussianBlur(transforms.RandomApply):
    """Randomly apply Gaussian blur to the image."""

    def __init__(self, *, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianBlur()
        super().__init__([transform], p=keep_p)


class RandomGaussianNoise(transforms.RandomApply):
    """Randomly apply Gaussian noise to the image."""

    def __init__(self, *, im_dim=144, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianNoise(im_dim=im_dim)
        super().__init__([transform], p=keep_p)
        
class GaussianNoise:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        mean: float = 0,
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2264926, 0.2431146, 0.1334844])
        self.loc_dist = np.array([-0.0006735, -0.0023663, -0.0143416])
        self.scale_dist = np.array([0.0037602, 0.0067417, 0.0260779])

        self.sigma_dist = np.log(self.scale_dist)

        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.001094, 0.001094, 0.001094])
        self.noise_ch_max = np.array([0.013, 0.018, 0.061])

    def __call__(self, image: np.ndarray):
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] += np.random.normal(
                    self.mean, self.sigma_augment[i], size=(self.im_dim, self.im_dim)
                )

        return image


class GaussianBlur:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2109966, 0.3008485, 0.3471172])
        self.loc_dist = np.array([1.0807153, 1.2394326, 1.1928363])
        self.scale_dist = np.array([1.3153171, 0.9164757, 0.8233702])

        self.sigma_dist = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3233109, 1.2667341, 1.2126263])
        self.psf_ch_max = np.array([5.0, 4.5, 4.25])

    def __call__(self, image: np.ndarray):
        # noise in channels is uncorrelated, as images taken at different times/telescopes
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] = skimage.filters.gaussian(
                    image[i, :, :], sigma=self.sigma_augment[i], mode="reflect"
                )

        return image