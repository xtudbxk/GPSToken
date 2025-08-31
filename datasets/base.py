import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

class ImagePaths(Dataset):
    def __init__(self, paths_or_file, size, random_crop=False, hflip=True):
        self.size = size
        self.random_crop = random_crop
        self.hflip = hflip

        if isinstance(paths_or_file, str):
            with open(paths_or_file, "r") as f:
                paths = [_l.strip("\n") for _l in f.readlines()]
        else:
            paths = paths_or_file
        self._length = len(paths)
        self.paths = paths

        self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
        if not self.random_crop:
            self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
        if self.hflip:
            self.flip = albumentations.HorizontalFlip(p=0.5)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper, self.flip])
        else:
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.paths[i])
        example["path"] = self.paths[i]
        return example
