"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


#CONSTATNS
REAL_LABEL = 0
FAKE_LABEL = 1
IMG_NAME_INDEX = 0
IMG_LABEL_INDEX = 1
REAL_DIRECTORY = 'real'
FAKE_DIRECTORY = 'fake'


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))        
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self._real_image_size = len(self.real_image_names)
        self._fake_image_size = len(self.fake_image_names)
        self._images = self.fake_image_names + self.real_image_names               
        self._dataset = list(zip(self._images, [FAKE_LABEL] * self._fake_image_size + [REAL_LABEL] * self._real_image_size))
        self._dataset_len = len(self._dataset)                        
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset.""" 
        img_name = self._dataset[index][IMG_NAME_INDEX]
        img_label = self._dataset[index][IMG_LABEL_INDEX]        
        if img_label == FAKE_LABEL:            
            img_path = os.path.join(self.root_path, FAKE_DIRECTORY, img_name)        
        else:
            img_path = os.path.join(self.root_path, REAL_DIRECTORY, img_name)        
        with Image.open(img_path) as im:
                if self.transform:
                    sample = self.transform(im)
        return (sample, img_label)
                     

    def __len__(self):
        """Return the number of images in the dataset."""        
        return self._dataset_len        
