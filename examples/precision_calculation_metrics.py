import torch
import torchmetrics
from torchvision.ops import nms


def post_process_yolo(predictions, confidence_threshold=0.5, nms_threshold=0.4):
    def xywh_to_xyxy(boxes):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        boxes[:, 0] = x - (w / 2)
        boxes[:, 1] = y - (h / 2)
        boxes[:, 2] = x + (w / 2)
        boxes[:, 3] = y + (h / 2)

    class_ids = []
    confidences = []
    boxes = []

    for prediction in predictions:
        prediction = prediction.squeeze()
        num_classes = prediction.shape[1] - 5

        objectness_scores = prediction[:, 4]
        class_scores = prediction[:, 5:]
        scores = objectness_scores.unsqueeze(1) * class_scores
        max_scores, class_ids = torch.max(scores, dim=1)
        mask = max_scores > confidence_threshold

        filtered_predictions = prediction[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if filtered_predictions.size(0) > 0:
            box = filtered_predictions[:, :4]
            xywh_to_xyxy(box)

            class_ids = class_ids.cpu().numpy()
            confidences = max_scores.cpu().numpy()
            boxes = box.cpu().numpy()

    if len(boxes) > 0:
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(confidences)
        class_ids = torch.from_numpy(class_ids)

        keep = nms(boxes, scores, nms_threshold)
        keep = keep.cpu().numpy()

        boxes = boxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

    return boxes, confidences, class_ids


def yolo_calculate_accuracy(res_list: list, iou_threshold=0.5):
    def calculate_iou(box1, box2):
        # Calculate the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate union area
        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union = area_box1 + area_box2 - intersection

        # Calculate IoU
        iou = intersection / union

        return iou

    pred_list = []
    target_list = []
    for item in res_list:
        pred_list.append(item[0])
        target_list.append(item[1])

    predictions_info = pred_list
    targets_info = target_list
    pred_boxes, pred_class_ids = predictions_info
    gt_boxes, gt_class_ids = targets_info

    total_predictions = len(pred_boxes)
    total_ground_truths = len(gt_boxes)
    correct_predictions = 0

    for i in range(total_predictions):
        pred_box = pred_boxes[i]
        pred_class_id = pred_class_ids[i]
        best_iou = 0.0
        best_gt_idx = -1

        for j in range(total_ground_truths):
            gt_box = gt_boxes[j]
            gt_class_id = gt_class_ids[j]

            if pred_class_id == gt_class_id:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx != -1:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def unet_accuracy_metric(res_list: list) -> float:
    def calculate_accuracy(pair):
        output = pair[0]
        target = pair[1]
        intersection = torch.logical_and(output, target).sum()
        union = torch.logical_or(output, target).sum()
        iou = intersection / (union + 1e-7)
        return iou

    metric_res = 0
    for res_pair in res_list:
        metric_res += calculate_accuracy(res_pair)
    avg_precision = metric_res / len(res_list)
    return avg_precision


def vgg16_accuracy_metric(res_list: list) -> float:
    return 0.0


def classification_top1_accuracy_metric(res_list: list) -> float:
    """
    Precision estimation based on torch metric.
    Run the metric on the iterable output.
    :param res_list: list containing pairs of predictions and targets.
    :return: float value of precision estimation
    """
    metric_res = 0
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=res_list[0][0].shape[1], top_k=1
    )
    for idx, res_pair in enumerate(res_list):
        if idx % 20 == 0:
            print(
                f"Calculating accuracy on prediction-target pair {idx}/{len(res_list)}"
            )
        metric_res += accuracy(res_pair[0], res_pair[1]).item()
    avg_precision = metric_res / len(res_list)
    return avg_precision


def classification_top5_accuracy_metric(res_list: list) -> float:
    """
    Precision estimation based on torch metric.
    Run the metric on the iterable output.
    :param res_list: list containing pairs of predictions and targets.
    :return: float value of precision estimation
    """
    metric_res = 0
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=res_list[0][0].shape[1], top_k=5
    )
    for idx, res_pair in enumerate(res_list):
        if idx % 20 == 0:
            print(
                f"Calculating accuracy on prediction-target pair {idx}/{len(res_list)}"
            )
        metric_res += accuracy(res_pair[0], res_pair[1]).item()
    avg_precision = metric_res / len(res_list)
    return avg_precision
