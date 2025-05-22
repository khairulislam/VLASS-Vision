from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import pickle, os
from urllib.request import urlopen
from sklearn.model_selection import train_test_split

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
        image = self.x[idx]
        target = self.y[idx]
        
        if self.transform is not None:
            image, target = self.transforms(image, target)
        
        return image, target
        
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