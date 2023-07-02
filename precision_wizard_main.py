import argparse
from precision_wizard import PrecisionWizard

# Supported models examples and parameters
from examples.supported_models import ModelParams, onnx_models_and_parameters

# Precision metrics examples
from examples.precision_calculation_metrics import (
    yolo_calculate_accuracy,
    unet_accuracy_metric,
    vgg16_accuracy_metric,
    classification_top1_accuracy_metric,
    classification_top5_accuracy_metric,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no_log", action="store_true", help="If used, logging is disabled."
)
parser.add_argument(
    "--use_gpu", action="store_true", help="If used, running Optimizer on GPU."
)
parser.add_argument(
    "--model_name",
    default="mobilenetv2_eval",
    help="Model name to use, taken from supported_models.onnx_models_and_parameters",
)
parser.add_argument(
    "--single_run",
    default=True,
    help="True: single model run | False: multiple models run",
)
parser.add_argument(
    "--dataset",
    default=None,
    help="Dataset to use, default taken from supported_models.onnx_models_and_parameters",
)

args = parser.parse_args()
is_log_disabled = args.no_log
use_gpu = args.use_gpu
single_run = args.single_run
model_name = args.model_name
dataset = args.dataset


if __name__ == "__main__":
    # test data
    total_networks = 0
    failed_networks_counter = 0
    # precision wizard config path, relative to working directory
    precision_wizard_config_file = "precision_wizard_config.yaml"
    # Initialize the PrecisionWizard
    precision_wizard = PrecisionWizard()
    if single_run is True:
        total_networks = 1
        model_params = onnx_models_and_parameters[model_name]
        if dataset is not None:
            model_params[-1] = dataset
        model_params_class = ModelParams(*model_params, use_gpu=use_gpu)
        # example of creating the Quantizer_List.txt file
        precision_wizard.generate_optimizer_layers_list(
            onnx_model=model_params_class.onnx_model,
            custom_tracer=model_params_class.custom_tracer,
            data_loaders=model_params_class.data_loaders,
            loader_key=model_params_class.loader_key,
        )
        print(f"Running on {model_name}")
        # Initialize the PrecisionWizard
        precision_wizard.init_tools(
            general_config_file=precision_wizard_config_file,
            onnx_model=model_params_class.onnx_model,
            data_loaders=model_params_class.data_loaders,
            loader_key=model_params_class.loader_key,
            custom_tracer=model_params_class.custom_tracer,
            use_gpu=use_gpu,
        )
        # unet not supporting top5 accuracy, replace accuracy metric
        if "unet_224" in model_params_class.onnx_model:
            precision_wizard.replace_precision_metric(
                metrics_list=[unet_accuracy_metric]
            )
        elif "vgg16" in model_params_class.onnx_model:
            precision_wizard.replace_precision_metric(
                metrics_list=[vgg16_accuracy_metric]
            )
        elif "yolo" in model_params_class.onnx_model:
            precision_wizard.replace_precision_metric(
                metrics_list=[yolo_calculate_accuracy]
            )
        elif "mobilenetv2_eval" in model_params_class.onnx_model:
            precision_wizard.replace_precision_metric(
                metrics_list=[
                    classification_top1_accuracy_metric,
                    classification_top5_accuracy_metric,
                ]
            )
        elif "yolo" in model_params_class.onnx_model:
            precision_wizard.replace_precision_metric([yolo_calculate_accuracy])
        # PrecisionWizard complete evaluation
        failed_networks_counter += precision_wizard.evaluate()

    else:  # MULTI_RUN
        # PrecisionWizard complete evaluation
        for idx, onnx_model_val_list in enumerate(onnx_models_and_parameters.values()):
            total_networks += 1
            model_params = ModelParams(*onnx_model_val_list, use_gpu=use_gpu)
            print(f"Running on {model_params.onnx_model}")
            # init Optimizer + Simulator tools
            precision_wizard.init_tools(
                general_config_file=precision_wizard_config_file,
                onnx_model=model_params.onnx_model,
                data_loaders=model_params.data_loaders,
                loader_key=model_params.loader_key,
                custom_tracer=model_params.custom_tracer,
                use_gpu=use_gpu,
                first_init=True if idx == 0 else False,
            )
            # unet not supporting top5 accuracy, replace accuracy metric
            if "unet_224" in model_params.onnx_model:
                precision_wizard.replace_precision_metric(
                    metrics_list=[unet_accuracy_metric]
                )
            elif "vgg16" in model_params.onnx_model:
                precision_wizard.replace_precision_metric(
                    metrics_list=[vgg16_accuracy_metric]
                )
            elif "yolo" in model_params.onnx_model:
                precision_wizard.replace_precision_metric(
                    metrics_list=[yolo_calculate_accuracy]
                )
            elif "mobilenetv2_eval" in model_params.onnx_model:
                precision_wizard.replace_precision_metric(
                    metrics_list=[
                        classification_top1_accuracy_metric,
                        classification_top5_accuracy_metric,
                    ]
                )
            # end-to-end evaluation
            failed_networks_counter += precision_wizard.evaluate()

    if failed_networks_counter == 0:
        print(
            f"All networks completed successfully, amount of networks: {total_networks}."
        )
    else:
        print(
            f"Successfully completed {total_networks-failed_networks_counter} out of {total_networks} networks."
        )
