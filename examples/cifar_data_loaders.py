# CIFAR100 data loaders example
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.models as models
import torchvision.models.resnet
import torchvision.datasets as torch_datasets
from torchvision.models.resnet import conv3x3, conv1x1
import torch.nn as nn
from torch import Tensor
import torch.onnx
from typing import Callable, Optional
import sys


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


torchvision.models.resnet.BasicBlock = BasicBlock
torchvision.models.resnet.Bottleneck = Bottleneck

# prepare both torch and onnx models of resnet18, used for pytest only
if any("pytest" in arg for arg in sys.argv):
    resnet18_torch_model = models.__dict__["resnet18"](pretrained=True)
    resnet18_onnx_model_name = "{0}.onnx".format(
        resnet18_torch_model.__class__.__name__
    )
    torch.onnx.export(
        resnet18_torch_model,
        torch.randn(1, 3, 224, 224),
        resnet18_onnx_model_name,
    )


class DataLoadersCifar100Example:
    @staticmethod
    def loader_key_function(use_gpu: bool = False):
        if use_gpu is True:
            return lambda model, x: model(x[0].cuda())
        return lambda model, x: model(x[0].cpu())

    @staticmethod
    def create_loaders(
        resize_size: int = 256, crop_size: int = 224, batch_size: int = 125
    ):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((crop_size, crop_size)),  # up-sample
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_valid = transforms.Compose(
            [
                transforms.Resize((crop_size, crop_size)),  # up-sample
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        dataset_folder = "/datasets/cifar"
        train_dataset = torch_datasets.CIFAR100(
            dataset_folder,
            train=True,
            transform=transform_train,
            target_transform=None,
            download=False,
        )
        test_dataset = torch_datasets.CIFAR100(
            dataset_folder,
            train=False,
            transform=transform_valid,
            target_transform=None,
            download=False,
        )

        calib = list(range(0, int(0.01 * len(train_dataset))))
        calibration_dataset = torch.utils.data.Subset(train_dataset, calib)
        calibration_dataset.transform = transforms.Compose(
            [
                transforms.Resize((crop_size, crop_size)),  # Upsample
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

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
        validation_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch,
            shuffle=test_sampler is None,
            num_workers=0,
            pin_memory=True,
            sampler=test_sampler,
        )

        return calibration_loader, validation_loader
