from examples.custom_tracers import *
from examples.imagenet_data_loaders import DataLoadersImagenet1000Example
from examples.cifar_data_loaders import DataLoadersCifar100Example
from examples.coco_data_loaders import DataLoadersCocoExample

# model and parameters of struct:
#   {
#   onnx_model_path,
#   resize_size,
#   crop_size,
#   batch_size,
#   custom_tracer
#   }


class ModelParams:
    def __init__(
        self,
        onnx_model,
        resize_size,
        crop_size,
        batch_size,
        custom_tracer,
        dataset,
        use_gpu,
    ):
        self.onnx_model = onnx_model
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.custom_tracer = custom_tracer
        self.dataset = dataset
        self.data_loaders = None
        self.loader_key = None
        self.create_data_loaders(use_gpu)

    def create_data_loaders(self, use_gpu):
        if "imagenet" in self.dataset.lower():
            self.data_loaders = DataLoadersImagenet1000Example.create_loaders(
                resize_size=self.resize_size,
                crop_size=self.crop_size,
                batch_size=self.batch_size,
            )
            self.loader_key = DataLoadersImagenet1000Example.loader_key_function(
                use_gpu
            )
        elif "cifar" in self.dataset.lower():
            self.data_loaders = DataLoadersCifar100Example.create_loaders(
                resize_size=self.resize_size,
                crop_size=self.crop_size,
                batch_size=self.batch_size,
            )
            self.loader_key = DataLoadersCifar100Example.loader_key_function(use_gpu)
        elif "coco" in self.dataset.lower():
            self.data_loaders = DataLoadersCocoExample.create_loaders(
                resize_size=self.resize_size,
                crop_size=self.crop_size,
                batch_size=self.batch_size,
            )
            self.loader_key = DataLoadersCocoExample.loader_key_function(use_gpu)
        else:
            # for other datasets, add your DataLoader class instantiation here
            print(f"Dataset {self.dataset} not supported")
            raise ValueError


onnx_models_and_parameters = {
    "inceptionv3_eval": [
        "/precision_wizard_networks/inceptionv3_eval.onnx",
        324,
        299,
        8,
        CustomTracerGeneral,
        "imagenet",
    ],
    "mobilenet_v1_1_224": [
        "/precision_wizard_networks/mobilenet_v1_1_224_no_transpose.onnx",
        256,
        224,
        1,
        CustomTracerMobilenetV1,
        "imagenet",
    ],
    "mobilenetv2_eval": [
        "/precision_wizard_networks/mobilenetv2_eval.onnx",
        256,
        224,
        8,
        None,
        "imagenet",
    ],
    "mobilenetv3_large_eval": [
        "/precision_wizard_networks/mobilenetv3_large_eval.onnx",
        256,
        224,
        8,
        None,
        "imagenet",
    ],
    "ResNet18": [
        "/precision_wizard_networks/ResNet18.onnx",
        256,
        224,
        8,
        None,
        "imagenet",
    ],
    "resnet50_custom_node": [
        "/precision_wizard_networks/resnet50_custom_node.onnx",
        256,
        224,
        1,
        CustomTracerGeneral,
        "imagenet",
    ],
    "unet_224": [
        "/precision_wizard_networks/unet_224.onnx",
        256,
        224,
        1,
        None,
        "imagenet",
    ],
    "vgg16": [
        "/precision_wizard_networks/vgg16.onnx",
        256,
        224,
        8,
        CustomTracerGeneral,
        "imagenet",
    ],
    "yolov3_no_post": [
        "/precision_wizard_networks/YOLO_V3_eval_no_post.onnx",
        416,
        416,
        8,
        CustomTracerYolo,
        "coco",
    ],
    "yolov4_no_post": [
        "/precision_wizard_networks/YOLO_V4_eval_no_post.onnx",
        416,
        416,
        8,
        CustomTracerYolo,
        "coco",
    ],
}
