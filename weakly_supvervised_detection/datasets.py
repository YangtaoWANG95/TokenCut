import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision import transforms


class CUB200(Dataset):
    def __init__(self, root, is_train, transform=None, ori_size=False, input_size=224, center_crop=True):
        self.root = root
        self.is_train = is_train
        self.ori_size = ori_size
        if not ori_size and center_crop: 
            image_size = int(256/224*input_size) #TODO check
            crop_size = input_size #TODO check
            shift = (image_size - crop_size) // 2
        elif not ori_size and not center_crop:
            image_size = input_size    
            crop_size = input_size
            shift = 0
        self.data = self._load_data(image_size, crop_size, shift, center_crop)
        self.transform = transform

    def _load_data(self, image_size, crop_size, shift, center_crop=True):
        self._labelmap_path = os.path.join(self.root, 'CUB_200_2011', 'classes.txt')

        paths = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'images.txt'),
            sep=' ', names=['id', 'path'])
        labels = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
            sep=' ', names=['id', 'label'])
        splits = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
            sep=' ', names=['id', 'is_train'])
        orig_image_sizes = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'image_sizes.txt'),
            sep=' ', names=['id', 'width', 'height'])
        bboxes = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'),
            sep=' ', names=['id', 'x', 'y', 'w', 'h'])
        
        if self.ori_size:
            resized_bboxes = pd.DataFrame({'id': paths.id,
                                           'xmin': bboxes.x,
                                           'ymin': bboxes.y,
                                           'xmax': bboxes.x + bboxes.w,
                                           'ymax': bboxes.y + bboxes.h})
        else:
            if center_crop:
                resized_xmin = np.maximum(
                    (bboxes.x / orig_image_sizes.width * image_size - shift).astype(int), 0)
                resized_ymin = np.maximum(
                    (bboxes.y / orig_image_sizes.height * image_size - shift).astype(int), 0)
                resized_xmax = np.minimum(
                    ((bboxes.x + bboxes.w - 1) / orig_image_sizes.width * image_size - shift).astype(int),
                    crop_size - 1)
                resized_ymax = np.minimum(
                    ((bboxes.y + bboxes.h - 1) / orig_image_sizes.height * image_size - shift).astype(int),
                    crop_size - 1)
            else:
                min_length = pd.concat([orig_image_sizes.width, orig_image_sizes.height], axis=1).min(axis=1)
                resized_xmin = (bboxes.x / min_length * image_size).astype(int)
                resized_ymin = (bboxes.y / min_length * image_size).astype(int)
                resized_xmax = ((bboxes.x + bboxes.w - 1) / min_length * image_size).astype(int)
                resized_ymax = ((bboxes.y + bboxes.h - 1) / min_length * image_size).astype(int)

            resized_bboxes = pd.DataFrame({'id': paths.id,
                                           'xmin': resized_xmin.values,
                                           'ymin': resized_ymin.values,
                                           'xmax': resized_xmax.values,
                                           'ymax': resized_ymax.values})

        
        data = paths.merge(labels, on='id')\
                    .merge(splits, on='id')\
                    .merge(resized_bboxes, on='id')

        if self.is_train:
            data = data[data.is_train == 1]
        else:
            data = data[data.is_train == 0]
        return data

    def __len__(self):
        return len(self.data)

 #   def _preprocess_bbox(self, origin_bbox, orig_image_size, center_crop=True):
 #       xmin, ymin, xmax, ymax = origin_bbox
 #       orig_width, orig_height = orig_image_size
 #       if center_crop:
 #           resized_xmin = np.maximum(
 #               (bboxes.x / orig_image_sizes.width * image_size - shift).astype(int), 0)
 #           resized_ymin = np.maximum(
 #               (bboxes.y / orig_image_sizes.height * image_size - shift).astype(int), 0)
 #           resized_xmax = np.minimum(
 #               ((bboxes.x + bboxes.w - 1) / orig_image_sizes.width * image_size - shift).astype(int),
 #               crop_size - 1)
 #           resized_ymax = np.minimum(
 #               ((bboxes.y + bboxes.h - 1) / orig_image_sizes.height * image_size - shift).astype(int),
 #               crop_size - 1)
 #       else:  
 #           print(f'width: {orig_image_sizes.width}, height: {orig_image_sizes.height}')
 #           min_length = min(orig_image_sizes.width , orig_image_sizes.height)
 #           resized_xmin = int(bb / min_length * self.image_size)
 #           resized_ymin = int(ymin / min_length * self.image_size)
 #           resized_xmax = int(xmax / min_length * self.image_size)
 #           resized_ymax = int(ymax / min_length * self.image_size)

 #           resized_bboxes = pd.DataFrame({'id': paths.id,
 #                                          'xmin': resized_xmin.values,
 #                                          'ymin': resized_ymin.values,
 #                                          'xmax': resized_xmax.values,
 #                                          'ymax': resized_ymax.values})

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, 'CUB_200_2011/images', sample.path)
        image = Image.open(path).convert('RGB')
        label = sample.label - 1 # label starts from 1
        gt_box = torch.tensor(
            [sample.xmin, sample.ymin, sample.xmax, sample.ymax])

        if self.transform is not None:
            image = self.transform(image)

        return (image, label, gt_box)

    @property
    def class_id_to_name(self):
        if hasattr(self, '_class_id_to_name'):
            return self._class_id_to_name
        labelmap = pd.read_csv(self._labelmap_path, sep=' ', names=['label', 'name'])
        labelmap['label'] = labelmap['label'].apply(lambda x: x - 1)
        self._class_id_to_name = labelmap.set_index('label')['name'].to_dict()
        return self._class_id_to_name

    @property
    def class_name_to_id(self):
        if hasattr(self, '_class_name_to_id'):
            return self._class_name_to_id
        self._class_name_to_id = {v: k for k, v in self.class_id_to_name.items()}
        return self._class_name_to_id

    @property
    def class_to_images(self):
        if hasattr(self, '_class_to_images'):
            return self._class_to_images
        #self.log.warn('Create index...')
        self._class_to_images = defaultdict(list)
        for idx in tqdm(range(len(self))):
            sample = self.data.iloc[idx]
            label = sample.label - 1
            self._class_to_images[label].append(idx)
        #self.log.warn('Done!')
        return self._class_to_images


#class ImageNet(H5Dataset):
#    def __init__(self, root, is_train, transform=None):
#        self.root = root
#        self.is_train = is_train
#        tag = 'train' if is_train else 'val'
#        self.h5_path = os.path.join(root, f'imagenet_{tag}.h5')
#
#        super().__init__(self.h5_path, transform)

class ImageNet(Dataset):
    def __init__(self, root, is_train, transform=None, ori_size=False, input_size=224, center_crop=True):
        self.root = root
        self.is_train = is_train
        self.ori_size = ori_size
        self.center_crop = center_crop
        if not ori_size and center_crop:
            self.image_size = int(256/224 * input_size) 
            self.crop_size = input_size
            self.shift = (self.image_size - self.crop_size) // 2
        elif not ori_size and not center_crop:
            print('resize, without center crop')
            self.image_size = input_size

        self._load_data()
        self.transform = transform

    def _load_data(self):
        self._labelmap_path = os.path.join(
            self.root, 'ILSVRC/Detection', 'imagenet1000_clsidx_to_labels.txt')

        if self.is_train:
            self.path = os.path.join(self.root, 'ILSVRC/Data/train')
            self.metadata = pd.read_csv(
                os.path.join(self.root, 'ILSVRC/Detection', 'train.txt'),
                sep=' ', names=['path', 'label'])
        else:
            self.path = os.path.join(self.root, 'ILSVRC/Data/val')
            self.metadata = pd.read_csv(
                os.path.join(self.root, 'ILSVRC/Detection', 'val.txt'),
                sep='\t', names=['path', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])
            self.wnids = pd.read_csv(
                os.path.join(self.root, 'ILSVRC/Detection/', 'wnids.txt'), names=['dir_name'])

    def _preprocess_bbox(self, origin_bbox, orig_image_size, center_crop=True, image_path=None):
        xmin, ymin, xmax, ymax = origin_bbox
        orig_width, orig_height = orig_image_size
        if center_crop:
            resized_xmin = np.maximum(
                int(xmin / orig_width * self.image_size - self.shift), 0)
            resized_ymin = np.maximum(
                int(ymin / orig_height * self.image_size - self.shift), 0)
            resized_xmax = np.minimum(
                int(xmax / orig_width * self.image_size - self.shift), self.crop_size - 1)
            resized_ymax = np.minimum(
                int(ymax / orig_height * self.image_size - self.shift), self.crop_size - 1)
        else:
            #print(f'ori W: {orig_width} ori H: {orig_height}, xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}, input_size: {self.image_size}, image_path: {image_path}')
            min_length = min(orig_height, orig_width)
            resized_xmin = int(xmin / min_length * self.image_size)
            resized_ymin = int(ymin / min_length * self.image_size)
            resized_xmax = int(xmax / min_length * self.image_size)
            resized_ymax = int(ymax / min_length * self.image_size)
            #print(f'output: xmin, ymin, xmax, ymax: {[resized_xmin, resized_ymin, resized_xmax, resized_ymax]}')
        return [resized_xmin, resized_ymin, resized_xmax, resized_ymax]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        if self.is_train:
            image_path = os.path.join(self.path, sample.path)
        else:
            image_path = os.path.join(
                self.path, self.wnids.iloc[int(sample.label)].dir_name, sample.path)
        image = Image.open(image_path).convert('RGB')
        label = sample.label

        # preprocess bbox
        if self.is_train:
            gt_box = torch.tensor([0., 0., 0., 0.])
        else:
            origin_box = [sample.xmin, sample.ymin, sample.xmax, sample.ymax]
            if self.ori_size:
                gt_box = torch.tensor(origin_box)
            else:
                gt_box = torch.tensor(
                    self._preprocess_bbox(origin_box, image.size, self.center_crop, image_path))

        if self.transform is not None:
            image = self.transform(image)

        return (image, label, gt_box)

    @property
    def class_id_to_name(self):
        if hasattr(self, '_class_id_to_name'):
            return self._class_id_to_name
        with open(self._labelmap_path, 'r') as f:
            self._class_id_to_name = eval(f.read())
        return self._class_id_to_name

    @property
    def class_name_to_id(self):
        if hasattr(self, '_class_name_to_id'):
            return self._class_name_to_id
        self._class_name_to_id = {v: k for k, v in self.class_id_to_name.items()}
        return self._class_name_to_id

    @property
    def wnid_list(self):
        if hasattr(self, '_wnid_list'):
            return self._wnid_list
        self._wnid_list = self.wnids.dir_name.tolist()
        return self._wnid_list

    @property
    def class_to_images(self):
        if hasattr(self, '_class_to_images'):
            return self._class_to_images
        self.log.warn('Create index...')
        self._class_to_images = defaultdict(list)
        for idx in tqdm(range(len(self))):
            sample = self.metadata.iloc[idx]
            label = sample.label
            self._class_to_images[label].append(idx)
        self.log.warn('Done!')
        return self._class_to_images

    def verify_wnid(self, wnid):
        is_valid = bool(re.match(u'^[n][0-9]{8}$', wnid))
        is_terminal = bool(wnid in self.wnids.dir_name.tolist())
        return is_valid and is_terminal

    def get_terminal_wnids(self, wnid):
        page = requests.get("http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={}&full=1".format(wnid))
        str_wnids = str(BeautifulSoup(page.content, 'html.parser'))
        split_wnids = re.split('\r\n-|\r\n', str_wnids)
        return [_wnid for _wnid in split_wnids if self.verify_wnid(_wnid)]

    def get_image_ids(self, wnid):
        terminal_wnids = self.get_terminal_wnids(wnid)

        image_ids = set()
        for terminal_wnid in terminal_wnids:
            class_id = self.wnid_list.index(terminal_wnid)
            image_ids |= set(self.class_to_images[class_id])

        return list(image_ids)

DATASETS = {
    'cub': CUB200,
    'imagenet': ImageNet,
}

LABELS = {
    'cub': 200,
    'imagenet': 1000,
}

def build_dataset(is_train, args):
    # Define arguments
    data_name = args.dataset
    root = args.data_path
    batch_size = args.batch_size_per_gpu
    num_workers = args.num_workers

    transform = build_transform(is_train, args)
    dataset = DATASETS[data_name](root, is_train=is_train, transform=transform, ori_size=args.ori_size, input_size = args.input_size, center_crop = not args.no_center_crop)

    return dataset, LABELS[data_name]

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and (not args.ori_size):
        if args.no_center_crop:
            t.append(transforms.Resize(args.input_size, interpolation=3))
        else:
            size = int((256 / 224) * args.input_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
    if not args.ori_size and not args.no_center_crop: 
        print('center crop')
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(t)
