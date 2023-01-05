import numpy as np
import pathlib
import cv2, random
import pandas as pd
import copy, json, os, math
import torch, glob
import pickle as pkl


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
                 dataset_type="train", balance_data=False, read_cross_dataset_path:str=None):
        if root:
            self.root = pathlib.Path(root)
        else:
            self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        self.read_cross_dataset_path = read_cross_dataset_path

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

    def _extract_file_name(self,df:pd.DataFrame)->list:
        s_ =[]
        for id in df['ImageID']:
            _, tail = os.path.split(id)
            s_.append(tail)
        return s_

    def _read_data(self):
        # class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        # class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
        class_dict = {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
        
        if self.dataset_type == 'train':
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons\TRAIN_objects2.csv"
        elif self.dataset_type == 'val':
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons\VAL_objects2.csv"
        elif self.dataset_type == 'test':
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons\TEST_objects2.csv"
        
        annotations = pd.read_csv(annotation_file)
        
        if self.read_cross_dataset_path:
            file_names = self._extract_file_name(annotations)
            annotations['FileName'] = file_names
            
            csv_files = []
            for path in glob.glob(os.path.join(self.read_cross_dataset_path,'*.csv')):
                csv_files.append(pd.read_csv(path))
            another_df = pd.concat(csv_files)
            
            file_names = self._extract_file_name(another_df)
            another_df['FileName'] = file_names
        
        data = []

        if self.read_cross_dataset_path:
            # another_df_group = another_df.groupby('FileName')
            # print(another_df_group.head(6))
            # print(len(another_df_group))

            print(another_df.head(5))
            print(len(another_df))
            for file_name, group in annotations.groupby("FileName"):
                file_match_index = another_df['FileName'] == file_name
                boxes = another_df.loc[file_match_index,["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
                image_ids = another_df.loc[file_match_index,["ImageID"]].values[0,0]
                labels = another_df.loc[file_match_index,["ClassName"]].values
                labels = labels[:,0]
                labels = np.array([class_dict[name] for name in labels], dtype='int64')
                assert len(labels) == len(boxes)
                data.append({
                    'image_id': image_ids,
                    'boxes': boxes,
                    'labels': labels
                })
            print('data read end')
            return data, class_names, class_dict
        else:

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


class MyImagesDataset:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False,
                 original_images_dir=None, crop_images_dir=None):
        assert original_images_dir and crop_images_dir
        if root:
            self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        self.original_images_dir = original_images_dir
        self.crop_images_dir = crop_images_dir

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()

        # self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _resize_and_rotate_with_corner_points(self,image:np.ndarray,corner_points:np.ndarray, init_size:int=300, final_size:int=300, rotate_degree=None)->'list[np.ndarray,np.ndarray]':
        h,w = image.shape[:2]
        print('image_shape:',image.shape)
        print('corner_points')
        print(corner_points)

        corner_points[:,::2] = corner_points[:,::2] / w
        corner_points[:,1::2] = corner_points[:,1::2] / h

        resized_image = cv2.resize(image,(init_size,init_size),interpolation=cv2.INTER_AREA)
        resized_h,resized_w = resized_image.shape[:2]

        print('resized_image_shape:',resized_image.shape)
        corner_points[:,::2] *= resized_w
        corner_points[:,1::2] *= resized_h
        print('resized_corner_points')
        print(corner_points)
        num_corner_points = len(corner_points)

        if not rotate_degree or rotate_degree == 0:
            ## boxes = ["XMin", "YMin", "XMax", "YMax"]
            boxes = np.zeros((num_corner_points,4))
            boxes[:,0] = np.min(corner_points[:,::2], axis=1)
            boxes[:,2] = np.max(corner_points[:,::2], axis=1)
            boxes[:,1] = np.min(corner_points[:,1::2], axis=1)
            boxes[:,3] = np.max(corner_points[:,1::2], axis=1)
            return resized_image, boxes

        c = np.cos(rotate_degree*np.pi/180)
        s = np.sin(rotate_degree*np.pi/180)
        rm = np.array([[c,-s],[s,c]])

        corner_points = corner_points.reshape(num_corner_points,-1,2)
        corner_points = corner_points.transpose(0,2,1)
        rotated_corner_points = np.matmul(rm,corner_points)
        print('rotated_corner_points: ', rotated_corner_points)

        boxes = np.zeros((num_corner_points,4))
        boxes[:,0] = np.min(rotated_corner_points[:,0,:], axis=1)
        boxes[:,2] = np.max(rotated_corner_points[:,0,:], axis=1)
        boxes[:,1] = np.min(rotated_corner_points[:,1,:], axis=1)
        boxes[:,3] = np.max(rotated_corner_points[:,1,:], axis=1)

        if rotate_degree == 90 or rotate_degree == -90:
            M = cv2.getRotationMatrix2D((resized_w/2,resized_h/2), rotate_degree,1)
            rotated_image = cv2.warpAffine(resized_image,M,(resized_w,resized_h))
            return rotated_image, boxes

        half_size = init_size//2
        point = np.array([[-half_size,half_size,half_size,-half_size],[half_size,half_size,-half_size,-half_size]])

        points = np.matmul(rm,point)
        min_x = min(points[0,:])
        max_x = max(points[0,:])
        min_y = max(points[1,:])
        max_y = max(points[1,:])
        left_pad = math.ceil(abs(abs(min_x)-half_size))
        right_pad = math.ceil(abs(abs(max_x)-half_size))
        top_pad = math.ceil(abs(abs(min_y)-half_size))
        bottom_pad = math.ceil(abs(abs(max_y)-half_size))
        print('prev_boxes')
        print(boxes)
        boxes[:,0] += left_pad
        boxes[:,2] += left_pad
        boxes[:,1] += top_pad
        boxes[:,3] += top_pad
        print('after_boxes')
        print(boxes)

        padded_image = np.pad(resized_image,((top_pad,bottom_pad),(left_pad,right_pad),(0,0)),'edge')
        padded_h,padded_w = padded_image.shape[:2]
        M = cv2.getRotationMatrix2D((padded_w/2,padded_h/2), rotate_degree,1)
        rotated_image = cv2.warpAffine(padded_image,M,(padded_w,padded_h))
        
        # M = cv2.getRotationMatrix2D((resized_w/2,resized_h/2), rotate_degree,1)
        # rotated_image = cv2.warpAffine(resized_image,M,(resized_w+left_pad+right_pad,resized_h+top_pad+bottom_pad))

        return rotated_image, boxes

    
    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        # duplicate boxes to prevent corruption of dataset
        # corner_points:np.ndarray = copy.copy(image_info['corner_points'])
        corner_points = image_info['corner_points'].astype(np.float64)
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        init_resize = 600
        init_resize = 300
        degree = 45
        image, boxes = self._resize_and_rotate_with_corner_points(image,corner_points,init_resize,rotate_degree=degree)

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
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons_pkl_datas\train_datas_25380_images.pkl"
        elif self.dataset_type == 'val':
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons_pkl_datas\val_datas_3172_images.pkl"
        elif self.dataset_type == 'test':
            annotation_file = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons_pkl_datas\test_datas_3173_images.pkl"
        
        with open(annotation_file,'rb') as f:
            annotations = pkl.load(f)
        
        class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
        class_dict = {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
        
        ori_datas = annotations['original_datas']
        crop_datas = annotations['crop_datas']

        image_ids = ori_datas['image_ids']
        corner_points = ori_datas['corner_points']
        labels = ori_datas['labels']
        difficulties = ori_datas['difficulties']
        
        image_files_dir = self.original_images_dir
        data = []
        for image_id, corner_point, label_list in zip(image_ids, corner_points, labels):
            ## corner_point:list[dict]
            ## [{'x1': ---, 'y1': ---, 'x2': ---, 'y2': ---, 'x3': ---, 'y3': ---, 'x4': ---, 'y4': ---},
            #   {'x1': ---, 'y1': ---, 'x2': ---, 'y2': ---, 'x3': ---, 'y3': ---, 'x4': ---, 'y4': ---},
            #   {'x1': ---, 'y1': ---, 'x2': ---, 'y2': ---, 'x3': ---, 'y3': ---, 'x4': ---, 'y4': ---},
            #   {'x1': ---, 'y1': ---, 'x2': ---, 'y2': ---, 'x3': ---, 'y3': ---, 'x4': ---, 'y4': ---},
            #   ... ]
            corners = []
            for dict_ in corner_point:
                corners.append([dict_['x1'],dict_['y1'],dict_['x2'],dict_['y2'],dict_['x3'],dict_['y3'],dict_['x4'],dict_['y4']])
            corners = np.array(corners,dtype=np.float64)
            data.append({'image_id': os.path.join(image_files_dir,image_id),
                        'corner_points':corners,
                        'labels':label_list})

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


if __name__ == '__main__':
    # datasets = MyImagesDataset(root=None,
    #             original_images_dir=r"C:\kwoncy\projects\xcode-detection\datas\xcode-new-datas",
    #                             crop_images_dir=r"C:\kwoncy\projects\xcode-detection\datas\xcode-new-datas-with-small-and-crop-images-2")
    config = mobilenetv1_ssd_config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    datasets = OpenImagesDataset3(root=None,
                                transform=train_transform, target_transform=target_transform,
                                dataset_type='train',read_cross_dataset_path=r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\jsons_crop")
                
    id, image, boxes, labels = datasets._getitem(160)

    print('image.shape: ',image.shape)
    print('boxes: ',boxes)
    print('labels: ',labels)
    distance = 5
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        image[y_min-5:y_min+5,x_min-5:x_min+5,:] = np.array([255,0,0])
        image[y_max-5:y_max+5,x_max-5:x_max+5,:] = [0,0,255]
    
    cv2.imshow(f'{id}',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()