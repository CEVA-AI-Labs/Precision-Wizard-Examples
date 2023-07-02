import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json


class CocoDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None):
        self.data_dir = data_dir
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)
        self.image_ids = [image["id"] for image in self.annotations["images"]]
        self.transform = transform

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = next(
            image for image in self.annotations["images"] if image["id"] == image_id
        )
        image_path = os.path.join(self.data_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id

    def __len__(self):
        return len(self.image_ids)


class DataLoadersCocoExample:
    @staticmethod
    def loader_key_function(use_gpu: bool = False):
        if use_gpu is True:
            return lambda model, x: model(x[0].cuda())
        return lambda model, x: model(x[0].cpu())

    @staticmethod
    def create_loaders(
        resize_size: int = 224,
        crop_size: int = 224,
        batch_size: int = 32,
    ):
        annotation_file = "/datasets/coco/annotations/person_keypoints_val2017.json"
        data_dir = "/datasets/coco/images/val2017"
        transform = transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        # Create the COCO dataset
        dataset = CocoDataset(data_dir, annotation_file, transform=transform)

        # Split the dataset into calibration and validation sets
        calibration_ratio = 0.8
        num_samples = len(dataset)
        calibration_size = int(calibration_ratio * num_samples)
        validation_size = num_samples - calibration_size

        calibration_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [calibration_size, validation_size]
        )

        # Create data loaders for calibration and validation
        calibration_batch_size = batch_size
        validation_batch_size = int(batch_size / 2)

        calibration_data_loader = DataLoader(
            calibration_dataset, batch_size=calibration_batch_size, shuffle=True
        )
        validation_data_loader = DataLoader(
            validation_dataset, batch_size=validation_batch_size, shuffle=False
        )

        return calibration_data_loader, validation_data_loader
