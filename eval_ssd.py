import torch, typing
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset, OpenImagesDataset3, MyImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse, os, time
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
import pandas as pd
import seaborn as sn
import shutil, re
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
    class_dict = {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
    label2class_dict = {1:"qrcode", 2:"barcode", 3:"mpcode", 4:"pdf417", 5:"dmtx", 0:"background"}

    true_case_stat = {}
    all_gt_boxes = {}
    gt_boxes_with_id_class = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if image_id not in gt_boxes_with_id_class:
                gt_boxes_with_id_class[image_id] = {}
            if class_index not in gt_boxes_with_id_class[image_id]:
                gt_boxes_with_id_class[image_id][class_index] = []
            gt_boxes_with_id_class[image_id][class_index].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for image_id in gt_boxes_with_id_class:
        for class_index in gt_boxes_with_id_class[image_id]:
            gt_boxes_with_id_class[image_id][class_index] = torch.stack(gt_boxes_with_id_class[image_id][class_index])
    
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            # all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
            selected = all_gt_boxes[class_index][image_id]
            all_gt_boxes[class_index][image_id] = selected.clone().detach()
    return true_case_stat, all_gt_boxes, all_difficult_cases, gt_boxes_with_id_class

def compute_confusion_matrix(true_case_stat, all_gt_boxes, all_difficult_cases,
                                        total_prediction_file, iou_threshold, use_2007_metric, gt_boxes_with_id_class):
    
    class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
    class_dict = {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
    label2class_dict = {1:"qrcode", 2:"barcode", 3:"mpcode", 4:"pdf417", 5:"dmtx", 0:"background"}

    with open(total_prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        n_labels = []
        labels = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[3]))
            box = torch.tensor([float(v) for v in t[4:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
            
            n_labels.append(int(float(t[1])))
            labels.append(t[2])
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        n_labels = [n_labels[i] for i in sorted_indexes]
        labels = [labels[i] for i in sorted_indexes]
        print('total predicted boxes: ',len(boxes))
        class_names.remove('background')
        cm = {class_name:{class_name2:0 for class_name2 in class_names+['background FN']+['total']} for class_name in class_names + ['background FP']}
        for key in cm.keys():
            cm[key]['FP_image_ids'] = []

        pred_boxes_with_id_class = {}
        ## make pred_boxes_with_id_class
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            n_label = n_labels[i]
            label = labels[i]

            if image_id not in pred_boxes_with_id_class:
                pred_boxes_with_id_class[image_id] = {}
            if n_label not in pred_boxes_with_id_class[image_id]:
                pred_boxes_with_id_class[image_id][n_label] = []
            pred_boxes_with_id_class[image_id][n_label].append(box)
        for image_id in pred_boxes_with_id_class:
            for n_label in pred_boxes_with_id_class[image_id]:
                pred_boxes_with_id_class[image_id][n_label] = torch.stack(pred_boxes_with_id_class[image_id][n_label])

        matched_infos = set()
        ## first, match True positive boxes
        for image_id in gt_boxes_with_id_class:
            for n_label in gt_boxes_with_id_class[image_id]:
                for i, selected_box in enumerate(gt_boxes_with_id_class[image_id][n_label]):
                    true_label = label2class_dict[n_label]
                    if image_id not in pred_boxes_with_id_class:
                        cm[true_label]['background FN'] += 1
                        cm[true_label]['total'] += 1
                        cm[true_label]['FP_image_ids'].append(image_id)
                        continue
                    if n_label not in pred_boxes_with_id_class[image_id]:
                        cm[true_label]['background FN'] += 1
                        cm[true_label]['total'] += 1
                        cm[true_label]['FP_image_ids'].append(image_id)
                        continue

                    pred_boxes = pred_boxes_with_id_class[image_id][n_label]
                    if len(pred_boxes) == 0:
                        cm[true_label]['background FN'] += 1
                        cm[true_label]['total'] += 1
                        cm[true_label]['FP_image_ids'].append(image_id)
                        continue
                    ious = box_utils.iou_of(selected_box,pred_boxes)
                    max_iou = torch.max(ious).item()
                    max_arg = torch.argmax(ious).item()
                    if max_iou > iou_threshold:
                        new_indices = torch.ones(len(ious)) == 1
                        new_indices[max_arg] = False
                        pred_boxes_with_id_class[image_id][n_label] = pred_boxes[new_indices]
                        cm[true_label][true_label] += 1
                        cm[true_label]['total'] += 1

                        matched_infos.add((image_id,n_label,i))
                    

        ## second, match False Positive and background FN 
        ## check remain pred_boxes with unmatched gt_boxes.
        for image_id in gt_boxes_with_id_class:
            for n_label in gt_boxes_with_id_class[image_id]:
                for i, selected_box in enumerate(gt_boxes_with_id_class[image_id][n_label]):
                    true_label = label2class_dict[n_label]
                    if image_id not in pred_boxes_with_id_class:
                        continue
                    if n_label not in pred_boxes_with_id_class[image_id]:
                        continue
                    if (image_id,n_label,i) in matched_infos:
                        continue

                    temp_max_iou=0
                    temp_max_arg=0
                    temp_class = None
                    temp_iou_len = 0
                    for temp_label in pred_boxes_with_id_class[image_id]:
                        temp_pred_boxes = pred_boxes_with_id_class[image_id][temp_label]
                        if len(temp_pred_boxes) == 0:
                            continue
                        ious = box_utils.iou_of(selected_box,temp_pred_boxes)
                        max_iou = torch.max(ious).item()
                        max_arg = torch.argmax(ious).item()

                        if max_iou > temp_max_iou:
                            temp_max_iou = max_iou
                            temp_max_arg = max_arg
                            temp_class = temp_label
                            temp_iou_len = len(ious)

                    if temp_max_iou > iou_threshold:
                        new_indices = torch.ones(temp_iou_len) == 1
                        new_indices[temp_max_arg] = False
                        temp_pred_boxes = pred_boxes_with_id_class[image_id][temp_class]
                        pred_boxes_with_id_class[image_id][temp_class] = temp_pred_boxes[new_indices]
                        
                        cm[true_label][label2class_dict[temp_class]] += 1
                        cm[true_label]['total'] += 1
                        cm[true_label]['FP_image_ids'].append(image_id)
                    else:
                        cm[true_label]['background FN'] += 1
                        cm[true_label]['total'] += 1
                        cm[true_label]['FP_image_ids'].append(image_id)

        ##third, check remain pred_boxes for background FP
        for image_id in pred_boxes_with_id_class:
            for n_label in pred_boxes_with_id_class[image_id]:
                pred_boxes = pred_boxes_with_id_class[image_id][n_label]
                pred_boxes_len = len(pred_boxes)
                if pred_boxes_len == 0:
                    continue
                cm['background FP'][label2class_dict[n_label]] += pred_boxes_len
                cm['background FP']['total'] += pred_boxes_len
                cm['background FP']['FP_image_ids'].append(image_id)

        for key in cm.keys():
            cm[key]['FP_image_ids'] = list(set(cm[key]['FP_image_ids']))
        
    return cm
    
def extract_fp_images(cm):
    extract_path = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\extract_fp_images"
    original_data_path = r"C:\kwoncy\projects\xcode-detection\datas\xcode-new-datas"
    crop_data_path = r"C:\kwoncy\projects\xcode-detection\datas\xcode-new-datas-with-small-and-crop-images-2"
    detected_data_path = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\draw_eval"
    whole_images = []
    for true_label in cm:
        if true_label == 'background FP':
            continue
        cur_true_label_path = os.path.join(extract_path,true_label)
        if not os.path.exists(cur_true_label_path):
            os.mkdir(cur_true_label_path)
        fp_image_paths = cm[true_label]['FP_image_ids']
        whole_images += fp_image_paths
        for image_path in fp_image_paths:
            ## image_path has cropped image path with no detected box.
            _, image_file_name = os.path.split(image_path)
            image_file_name_wo_ext, ext = os.path.splitext(image_file_name)

            shutil.copy(os.path.join(original_data_path,image_file_name),os.path.join(cur_true_label_path,f'{image_file_name_wo_ext}_1_original{ext}'))
            shutil.copy(image_path,os.path.join(cur_true_label_path,f'{image_file_name_wo_ext}_2_crop{ext}'))
            new_image_path = image_path.replace('old','new')
            shutil.copy(new_image_path,os.path.join(cur_true_label_path,f'{image_file_name_wo_ext}_3_crop_label{ext}'))
            txt_file_name = re.sub(f"-(new|old)_img{ext}",".txt",image_file_name)
            crop_label_txt_path = os.path.join(crop_data_path,txt_file_name)
            shutil.copy(crop_label_txt_path,os.path.join(cur_true_label_path,f'{image_file_name_wo_ext}_3_crop_label.txt'))
            shutil.copy(os.path.join(detected_data_path,image_file_name),os.path.join(cur_true_label_path,f'{image_file_name_wo_ext}_4_detected{ext}'))
    
    whole_images = list(set(whole_images))
    total_folder_path = os.path.join(extract_path,"total") 
    if not os.path.exists(total_folder_path):
        os.mkdir(total_folder_path)

    for image_path in whole_images:
        ## image_path has cropped image path with no detected box.
        _, image_file_name = os.path.split(image_path)
        image_file_name_wo_ext, ext = os.path.splitext(image_file_name)

        shutil.copy(os.path.join(original_data_path,image_file_name),os.path.join(total_folder_path,f'{image_file_name_wo_ext}_1_original{ext}'))
        shutil.copy(image_path,os.path.join(total_folder_path,f'{image_file_name_wo_ext}_2_crop{ext}'))
        new_image_path = image_path.replace('old','new')
        shutil.copy(new_image_path,os.path.join(total_folder_path,f'{image_file_name_wo_ext}_3_crop_label{ext}'))
        txt_file_name = re.sub(f"-(new|old)_img{ext}",".txt",image_file_name)
        crop_label_txt_path = os.path.join(crop_data_path,txt_file_name)
        shutil.copy(crop_label_txt_path,os.path.join(total_folder_path,f'{image_file_name_wo_ext}_3_crop_label.txt'))
        shutil.copy(os.path.join(detected_data_path,image_file_name),os.path.join(total_folder_path,f'{image_file_name_wo_ext}_4_detected{ext}'))
    
    return


def make_cm2plot(cm, save_dir='',names=()):
    n_cm = len(cm)
    max_index = n_cm-1 
    names = [k for k in cm.keys() if k not in ['total','FP_image_ids']]
    print(names)
    matrix = np.zeros((n_cm,n_cm))
    for i, k in enumerate(cm.keys()):
        for j, sub_k in enumerate(cm[k].keys()):
            if 'total' in sub_k or 'FP_image_ids' in sub_k:
                continue
            # matrix[max_index-j,i] = cm[k][sub_k]
            matrix[j,i] = cm[k][sub_k]
            
    array = matrix / (matrix.sum(0).reshape(1,n_cm) + 1e-6)
    array[array < 0.005] = np.nan

    fig = plt.figure(figsize=(12,9), tight_layout=True)
    sn.set(font_scale=1.0 if n_cm < 50 else 0.8)
    labels = (0<len(names)<99) and len(names) == n_cm
    names2 = [v for v in cm[names[0]].keys() if v not in ['total','FP_image_ids']]
    sn.heatmap(array, annot=n_cm<30, annot_kws={"size":8}, cmap='Blues', fmt='.2f', square=True,
                xticklabels=names if labels else "auto",
                yticklabels=names2 if labels else "auto").set_facecolor((1,1,1))
    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    fig.savefig('confusion_matrix.png', dpi=250)

    # array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
    # array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    # fig = plt.figure(figsize=(12, 9), tight_layout=True)
    # sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
    # labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
    # sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
    #             xticklabels=names + ['background FP'] if labels else "auto",
    #             yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
    # fig.axes[0].set_xlabel('True')
    # fig.axes[0].set_ylabel('Predicted')
    # fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
    # return




    #     for i, image_id in enumerate(image_ids):
    #         box = boxes[i]
    #         n_label = n_labels[i]
    #         label = labels[i]

    #         if image_id not in gt_boxes_with_id_class:
    #             cm['background FP'][label] += 1
    #             cm['background FP']['total'] += 1
    #             continue
    #         if n_label not in gt_boxes_with_id_class[image_id]:
    #             cm['background FP'][label] += 1
    #             cm['background FP']['total'] += 1
    #             continue

    #         gt_boxes = gt_boxes_with_id_class[image_id][n_label]
    #         gt_boxes_len = len(gt_boxes)
    #         if gt_boxes_len == 0:
    #             cm['background FP'][label] += 1
    #             cm['background FP']['total'] += 1
    #             continue

    #         ious = box_utils.iou_of(box, gt_boxes)
    #         max_iou = torch.max(ious).item()
    #         max_arg = torch.argmax(ious).item()
    #         flag_cur_box_matched = True
    #         if max_iou > iou_threshold:
    #             cm[label][label] += 1
    #             cm[label]['total'] += 1
                
    #             new_indices = torch.ones(len(ious)) == 1
    #             new_indices[max_arg] = False
    #             gt_boxes_with_id_class[image_id][n_label] = gt_boxes[new_indices]
                
    #         else:
    #             temp_max_iou=0
    #             temp_max_arg=0
    #             temp_class=None
    #             temp_iou_len = 0
    #             for class_ in gt_boxes_with_id_class[image_id]:
    #                 class_gt_box = gt_boxes_with_id_class[image_id][class_]
    #                 if len(class_gt_box) == 0:
    #                     continue
    #                 ious = box_utils.iou_of(box, class_gt_box)
    #                 max_iou = torch.max(ious).item()
    #                 max_arg = torch.argmax(ious).item()
    #                 if max_iou > temp_max_iou:
    #                     temp_max_iou = max_iou
    #                     temp_max_arg = max_arg
    #                     temp_class = class_
    #                     temp_iou_len = len(ious)
    #             if temp_max_iou > iou_threshold:
    #                 cm[label][label2class_dict[temp_class]] += 1
    #                 cm[label]['total'] += 1
    #                 new_indices = torch.ones(temp_iou_len) == 1
    #                 new_indices[temp_max_arg] = False
    #                 class_gt_box = gt_boxes_with_id_class[image_id][temp_class]
    #                 gt_boxes_with_id_class[image_id][temp_class] = class_gt_box[new_indices]
    #             else:
    #                 cm[label]['background'] += 1
    #                 cm[label]['total'] += 1


    #     ## calculate remain boxes
    #     for id, class_with_boxes in gt_boxes_with_id_class.items():
    #         for class_ in class_with_boxes:
    #             box = class_with_boxes[class_]
    #             remain_box_count = len(box)
    #             if remain_box_count == 0:
    #                 continue
                
    #             # cm[class_]['backgroud'] += remain_box_count
    #             # cm[class_]['total'] += remain_box_count

    #             cm["background FP"][label2class_dict[class_]] += remain_box_count
    #             cm["background FP"]['total'] += remain_box_count

    # return cm

    

def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    _, name = os.path.split(prediction_file)
    # print(f'total box of {name} : ', len(boxes))
    # print('TP len :', len(true_positive))
    # print('FP len:', len(false_positive))

    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


def get_class_ap_dict(txt_path:str, dataset:typing.Union[OpenImagesDataset3, MyImagesDataset], class_names:list) -> dict:
    true_case_stat, all_gb_boxes, all_difficult_cases, _ = group_annotation_by_class(dataset)
    print("\n\nAverage Precision Per-class:")
    class_ap_dict = {}
    aps = []
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = os.path.join(txt_path,f"det_test_{class_name}.txt")
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        class_ap_dict[class_name] = ap
        aps.append(ap)
        print(f"{class_name}: {ap}")

    total_mAP = sum(aps)/len(aps)
    class_ap_dict['total'] = total_mAP 
    print(f"\nAverage Precision Across All Classes:{total_mAP}")

    return class_ap_dict


def make_predicted_txt(eval_path, dataset=None, class_names=None, model_state_dict=None, pretrained_model_path=None):
    # eval_path = pathlib.Path(args.eval_dir)
    
    # eval_path.mkdir(exist_ok=True)
    timer = Timer()

    class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
    class_dict = {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
    label2class_dict = {1:"qrcode", 2:"barcode", 3:"mpcode", 4:"pdf417", 5:"dmtx", 0:"background"}
    # args.dataset = os.path.join(os.getcwd(),'jsons')
    args.net = 'mb3-small-ssd-lite'
    args.net = 'mb1-ssd'
    args.net = 'mb2-ssd-lite'
    args.net = 'mb2-ssd600-lite'
    # args.trained_model = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\checkpoint\mb2-ssd-lite-Epoch-65-Loss-3.078031623363495.pth"
    args.trained_model = pretrained_model_path
    args.dataset_type = 'xcode'
    args.nms_method = 'hard'
    args.iou_threshold = 0.5
    args.mb2_width_mult = 1.0
    args.eval_dir = 'eval_results'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # args.net = 'mb3-small-ssd-lite'
    # args.label_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\checkpoint\voc199\voc-model-labels.txt"
    # class_names = [name.strip() for name in open(args.label_file).readlines()]
    # args.dataset_type = 'voc'
    # # args.datasets = ['D:\datas\VOCdevkit\VOC2007','D:\datas\VOCdevkit\VOC2012']
    # args.trained_model = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\checkpoint\mb3-small-ssd-lite-Epoch-199-Loss-4.12402254227669.pth"
    # args.dataset = r'D:\datas\VOCdevkit\VOC2007-test'
    # args.validation_dataset = r'D:\datas\VOCdevkit\VOC2007-test'
    # args.t_max = 200
    # args.validation_epochs = 5
    # args.num_epochs = 200
    # args.scheduler = 'cosine'
    # args.lr = 0.01


    # if args.dataset_type == "voc":
    #     dataset = VOCDataset(args.dataset, is_test=True)
    # elif args.dataset_type == 'open_images':
    #     dataset = OpenImagesDataset(args.dataset, dataset_type="test")
    # elif args.dataset_type == 'xcode':
    #     # dataset = OpenImagesDataset2(args.dataset, dataset_type="test")
    #     dataset = OpenImagesDataset3(args.dataset, dataset_type="test")
    #     # label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
    #     # store_labels(label_file, dataset.class_names)
    #     # logging.info(dataset)
    #     # num_classes = len(dataset.class_names)
    true_case_stat, all_gb_boxes, all_difficult_cases, _ = group_annotation_by_class(dataset)

    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'mb3-large-ssd-lite':
        net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb3-small-ssd-lite':
        net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd600-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, ssd600=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    if pretrained_model_path:
        # net.load(args.trained_model)
        net.load(pretrained_model_path)
    if model_state_dict:
        net.load_state_dict(model_state_dict)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite' or args.net == "mb3-large-ssd-lite" or args.net == "mb3-small-ssd-lite":
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd600-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE, ssd600=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    
    print(f"predict start with len(dataset): {len(dataset)}")
    results = []
    image_ids = []
    predict_time = time.time()
    for i in range(len(dataset)):
    # for i in range(100):
        # print("process image", i)

        timer.start("Load Image")
        if isinstance(dataset,MyImagesDataset):
            _, image, _, _ = dataset.get_image(i)
        else:
            image = dataset.get_image(i)
        cur_data = dataset.data[i]
        image_id = cur_data['image_id']
        
        # print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))

        empty = torch.tensor([])
        if boxes.size() == empty.size():
            print(f"empty detected on image_id({image_id})")
            continue
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        image_ids.extend([image_id] * len(indexes))


        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    print(f"predict done. len(dataset): {len(dataset)}, time: {time.time()-predict_time}")

    print(f"making eval_txt files start")
    make_txt_time = time.time()
    total_line = 0
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        # prediction_path = eval_path / f"det_test_{class_name}.txt"
        prediction_path = os.path.join(eval_path,f"det_test_{class_name}.txt")
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
                total_line += 1
    print(f"making eval_txt files done. total_line: {total_line}, time: {time.time()-make_txt_time}")

    print(f"making eval_total_txt files start")
    make_total_txt_time = time.time()
    df = pd.DataFrame(results.numpy(force=True),columns=['Dataset_index', 'Label', 'Prob', 'Xmin', 'Ymin', 'Xmax', 'Ymax'])
    df['Image_id'] = image_ids
    total_line = 0
    prediction_path = os.path.join(os.getcwd(),'eval_results','det_test_total.txt')
    with open(prediction_path, 'w') as f1:
        for img_id, g in df.groupby('Image_id'):
            g = g.sort_values('Prob')
            for row in g.itertuples():
                # if row.Prob < 0.5:
                    # continue
                label_ = row.Label
                class_ = label2class_dict[int(label_)]
                print(f'{img_id} {row.Label} {class_} {row.Prob} {row.Xmin} {row.Ymin} {row.Xmax} {row.Ymax}', file=f1)
                total_line += 1
    
    print(f"making eval_total_txt files done. total_line: {total_line}, time: {time.time()-make_total_txt_time}")

    print(f"making eval_total2_txt files start ( > 0.5)")
    df2 = df[df['Prob'] > 0.5]
    total_line = 0 
    make_total2_txt_time = time.time()
    prediction_path = os.path.join(os.getcwd(),'eval_results','det_test_total2.txt')
    with open(prediction_path, 'w') as f2:
        for img_id, g in df2.groupby('Image_id'):
            g = g.sort_values('Prob')
            for row in g.itertuples():
                # if row.Prob < 0.5:
                    # continue
                label_ = row.Label
                class_ = label2class_dict[int(label_)]
                print(f'{img_id} {row.Label} {class_} {row.Prob} {row.Xmin} {row.Ymin} {row.Xmax} {row.Ymax}', file=f2)
                total_line += 1

    print(f"making eval_total2_txt files done. total_line: {total_line}, time: {time.time()-make_total2_txt_time}")


if __name__ == '__main__':
    
    eval_path = os.path.join(os.getcwd(),'eval_results')
    # dataset = OpenImagesDataset3(os.path.join(os.getcwd(),'jsons'), dataset_type="test", read_cross_dataset_path=r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons_original")
    dataset = OpenImagesDataset3(os.path.join(os.getcwd(),'jsons'), dataset_type="test", read_cross_dataset_path=r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons_crop")
    # dataset = OpenImagesDataset3(os.path.join(os.getcwd(),'jsons'), dataset_type="test")
    class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
    pretrained = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\checkpoint\mb2-ssd-lite-Epoch-445-Loss-1.722874907851219.pth"
    make_predicted_txt(eval_path, dataset, class_names, None, pretrained)
    get_class_ap_dict(eval_path, dataset, class_names)
    # true_case_stat, all_gt_boxes, all_difficult_cases, gt_boxes_with_id_class = group_annotation_by_class(dataset)
    # cm = compute_confusion_matrix(true_case_stat, all_gt_boxes,all_difficult_cases, os.path.join(os.getcwd(),'eval_results','det_test_total2.txt'),0.5,False,gt_boxes_with_id_class)
    # cm2={}
    # for k,v in cm.items():
    #     if 'FP_image_ids' in v:
    #         del v['FP_image_ids']
    #     cm2[k] = v
    # print(cm2)
    # # extract_fp_images(cm)
    # make_cm2plot(cm2)