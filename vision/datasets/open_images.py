import numpy as np
import pathlib
import cv2
import pandas as pd
import copy, json, os
import torch

class OpenImagesDataset:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        annotation_file = f"{self.root}/sub-{self.dataset_type}-annotations-bbox.csv"
        annotations = pd.read_csv(annotation_file)
        class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            # make labels 64 bits to satisfy the cross_entropy function
            labels = np.array([class_dict[name] for name in group["ClassName"]], dtype='int64')
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_id):
        image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data



class OpenImagesDataset3:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        # image = cv2.resize(image,(300,300),interpolation=cv2.INTER_AREA)
        # image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        # image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        if self.dataset_type == 'train':
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons\TRAIN_objects2.csv"
        elif self.dataset_type == 'val':
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons\VAL_objects2.csv"
        elif self.dataset_type == 'test':
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons\TEST_objects2.csv"
        
        annotations = pd.read_csv(annotation_file)
        # class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        # class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
        class_dict = {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
        
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            # make labels 64 bits to satisfy the cross_entropy function
            labels = np.array([class_dict[name] for name in group["ClassName"]], dtype='int64')
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_id):
        # image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        # image = cv2.imread(str(image_file))
        image = cv2.imread(str(image_id))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data


class OpenImagesDataset2(OpenImagesDataset):
    def __init__(self, root, transform=None, target_transform=None,
                dataset_type="train", balance_data=False):
        super(OpenImagesDataset2,self).__init__(root, transform, target_transform,
                dataset_type, balance_data)
    
    def _getitem(self, index):
        image_info = self.data[index]
        # image = self._read_image(image_info['image_id'])
        image = self._read_image(image_info['image_path'])

        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])
        new_boxes = copy.copy(image_info['boxes'])
        # boxes[:, 0] *= image.shape[1]
        # boxes[:, 1] *= image.shape[0]
        # boxes[:, 2] *= image.shape[1]
        # boxes[:, 3] *= image.shape[0]
        new_boxes[:,1] = boxes[:,2]
        new_boxes[:,2] = boxes[:,1]
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        if self.transform:
            image, new_boxes, labels = self.transform(image, new_boxes, labels)
        if self.target_transform:
            new_boxes, labels = self.target_transform(new_boxes, labels)
        # return image_info['image_id'], image, boxes, labels
        return image_info['image_path'], image, new_boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels


    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_path'])
        if self.transform:
            image, _ = self.transform(image)
        return image
        
    def _read_data(self):
        dataset_type = self.dataset_type.upper()
        root = '.'
        with open(f"{root}/jsons/{dataset_type}_images.json", "r") as j:
            image_paths = json.load(j)
        with open(f"{root}/jsons/{dataset_type}_objects.json", "r") as j:
            annotations = json.load(j)
        with open(f"{root}/jsons/label_map.json","r") as j:
            label_map = json.load(j)

        data = []
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            annotation = annotations[i]
            head, tail = os.path.split(image_path)
            file_name, ext = os.path.splitext(tail)
            data.append({'image_path':image_path , \
                        'image_id': file_name, \
                        'boxes': np.array(annotation["boxes"], dtype=np.float64), \
                        'labels': np.array(annotation["labels"])})
        # class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        # class_names = list(label_map.keys())
        # class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
        class_dict = {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
        # class_dict = label_map
        # for image_id, group in annotations.groupby("ImageID"):
        #     boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
        #     # make labels 64 bits to satisfy the cross_entropy function
        #     labels = np.array([class_dict[name] for name in group["ClassName"]], dtype='int64')
        #     data.append({
        #         'image_id': image_id,
        #         'boxes': boxes,
        #         'labels': labels
        #     })
        return data, class_names, class_dict


    # def _read_image(self, image_id):
    def _read_image(self, image_path):
        # image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __repr__(self):
        if self.class_stat is None:
            # self.class_stat = {name: 0 for name in self.class_names[1:]}
            self.class_stat = {name: 0 for name in self.class_names}
            # self.index2name = {"0":'background', "1":"qrcode", "2":"barcode", "3":"mpcode", "4":"pdf417", "5":"dmtx"}
            self.index2name = {0:'background', 1:"qrcode", 2:"barcode", 3:"mpcode", 4:"pdf417", 5:"dmtx"}
            # {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
            for example in self.data:
                for class_index in example['labels']:
                    
                    # class_name = self.class_names[class_index]
                    class_name = self.index2name[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)
        
    
