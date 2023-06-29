import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as torch_datasets
from torchvision.transforms.functional import InterpolationMode
import torch.fx
import torch.onnx


def dumpToFolder(images, root):
    """
    This function generates smaller dataset with ground truth. Used to run CDNN
    :param images: images to save to folder
    :param root: where to save them
    :return:
    """
    from shutil import copyfile

    os.mkdir("images30")
    os.mkdir("generationDataBase")
    toRun = images[:5000]
    toGenerate = images[5000:5150]
    with open("groundTruth.txt", "w") as lab:
        for i in toRun:
            lab.write(i)
            name, label = i.split(" ")
            fullPath = os.path.join(root, name)
            copyfile(fullPath, "{}/{}".format("images30", name.split("/")[1]))
    for i in toGenerate:
        name, label = i.split(" ")
        fullPath = os.path.join(root, name)
        copyfile(fullPath, "{}/{}".format("generationDataBase", name.split("/")[1]))
    print("finished")


class ImageNetDataset(Dataset):
    def __init__(
        self,
        images_root,
        labels_path,
        classes_num=-1,
        labelsToUse=None,
        shuffle=True,
        transform=None,
        target_transform=None,
    ):
        """
        :param images_root: images folder to imagenet
        :param labels_path: labels folder for imagenet
        :param classes_num: number of classes to use
        :param labelsToUse: list of spesific classes to use
        :param shuffle: should randomly shuffle images
        :param transform: image transform
        :param target_transform: label transform
        """
        labelsToUse = None
        with open(labels_path) as lab:
            self.images = lab.readlines()
        self._labelsToUse = []
        if labelsToUse is not None and classes_num != -1:
            raise ValueError(
                " only one limiting parameter can be used, either class num or labelsToUse"
            )
        if classes_num != -1:
            allLabels = [int(image.split(" ")[1]) for image in self.images]
            allLabels = list(set(allLabels))
            labelsToUse = allLabels[:classes_num]
        self._labelsToUse = labelsToUse
        if self._labelsToUse is not None:
            self.images = [
                image
                for image in self.images
                if int(image.split(" ")[1]) in self._labelsToUse
            ]

        self.root = images_root
        if shuffle:
            random.shuffle(self.images)

        self.images = self.images
        self.transform = transform
        self.target_transform = target_transform
        # dumpToFolder(self.images, self.root)

    def getLabelsToUse(self):
        return self._labelsToUse

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        pickedGoodImage = False
        while not pickedGoodImage:
            image = self.images[index]
            name, label = image.split(" ")
            fullPath = os.path.join(self.root, name)
            if not os.path.exists(fullPath):
                index = index - 1 if index > 0 else index + 1
                print("image name: {} does not exist".format(fullPath))
                continue
            try:
                img = Image.open(fullPath)
            except:
                index = index - 1 if index > 0 else index + 1
                continue

            label = int(label)
            if not len(img.getbands()) == 3:
                index = index - 1 if index > 0 else index + 1
                continue
            try:
                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    label = self.target_transform(label)
            except Exception as err:
                print(f"transform failed on image {fullPath}, with error: {err}")
                index = index - 1 if index > 0 else index + 1
                pickedGoodImage = False
                continue
            return img, label


class DataLoadersImagenet1000Example:
    @staticmethod
    def loader_key_function(use_gpu: bool = False):
        if use_gpu is True:
            return lambda model, x: model(x[0].cuda())
        return lambda model, x: model(x[0].cpu())

    @staticmethod
    def create_loaders(
        resize_size: int = 256,
        crop_size: int = 224,
        batch_size: int = 125,
        shuffle=True,
    ):
        imagenet_labels = r"/datasets/ImageNet/train-labels.txt"
        imagenet_folder = r"/datasets/ImageNet/train"
        imagenet_labelsVal = r"/datasets/ImageNet/val-labels.txt"
        imagenet_folderVal = r"/datasets/ImageNet/val"

        interpolation = InterpolationMode.BILINEAR

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        preprocess_train = train_transform
        preprocess_test = test_transform
        classesToUse = 1000

        train_dataset = ImageNetDataset(
            images_root=imagenet_folder,
            labels_path=imagenet_labels,
            transform=preprocess_train,
            classes_num=classesToUse,
            shuffle=shuffle,
        )
        usedLabels = train_dataset.getLabelsToUse()
        test_dataset = ImageNetDataset(
            images_root=imagenet_folderVal,
            labels_path=imagenet_labelsVal,
            transform=preprocess_test,
            labelsToUse=usedLabels,
            shuffle=shuffle,
        )

        calib = list(range(0, int(0.001 * len(train_dataset))))
        calibration_dataset = torch.utils.data.Subset(train_dataset, calib)
        calibration_dataset.transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        classes = 1000

        calib_sampler = None
        test_sampler = None
        test_batch = batch_size

        calibration_loader = torch.utils.data.DataLoader(
            calibration_dataset,
            batch_size=test_batch,
            shuffle=calib_sampler is None,
            num_workers=0,
            pin_memory=True,
            sampler=calib_sampler,
        )
        test_dataset.images = test_dataset.images[:200]
        validation_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch,
            shuffle=test_sampler is None,
            num_workers=0,
            pin_memory=True,
            sampler=test_sampler,
        )

        return calibration_loader, validation_loader
