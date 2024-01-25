import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class ADE20KDataset(Dataset):
    """
        self.label_text_ADE: text label
        self.places_class_names: class names of places dataset, serving as a converter between ade class and palces class
        self.label_idx_PLACES: class indexes for places dataset
        root
        images: dir of each image
        masks: dir of each image mask
    """
    
    def __init__(self, 
                 dir_root="DataSet/ADEChallengeData2016", 
                 dir_assist_label="DataSet/categories_places365.txt",
                 split='train', 
                 **kwargs):
        self.root = dir_root
        assert os.path.exists(self.root), f"Error with the dataset path; make sure the data is found in the root {self.root}"
        
        self.images, self.masks = self._get_ade20k_pairs(self.root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))
        idx_fid = np.array([path.split('/')[-1].replace('.jpg', '') for path in self.images])
        img_label = self._get_classes(self.root+'/sceneCategories.txt') 
        self.label_text_ADE = np.array([img_label[fid] for fid in idx_fid]) 
        self._adapt_places(dir_assist_label)

    def __getitem__(self, index, return_msk = False):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]) if return_msk else None
        return (img, mask) if return_msk else img

    def __len__(self):
        return len(self.images)
    
    def filter_class(self, thresh: int) -> np.array:
        unique_cls, cls_cnt = np.unique(self.label_idx_PLACES, return_counts=True)
        return unique_cls[np.where(cls_cnt >= thresh)]
    
    def get_test_samples(self, thresh: int, class_count: int, sample_per_class: int) -> 'tuple[list,list]':
        img_ids = []
        img_lbs = []
        for cls in np.random.choice(self.filter_class(thresh), class_count, replace=False):
            for sid in np.random.choice(self.get_class_sample(cls), sample_per_class, replace=False):
                img_ids.append(sid)
                img_lbs.append(cls)
        return img_ids, img_lbs
    
    def get_class_sample(self, cls: int) -> np.array:
        return np.argwhere(self.label_idx_PLACES == cls).reshape(-1)

    def _get_classes(self, filename):
        img_label = {}
        with open(filename) as f:
            lines = f.read().splitlines()
            splitted_lines = np.array([line.split() for line in lines])
            for fid, label in zip(splitted_lines[:,0], splitted_lines[:,1]):
                img_label[fid] = label
        return img_label

    def _get_ade20k_pairs(self, folder, mode='train'):
        img_paths = []
        mask_paths = []
        if mode == 'train':
            img_folder = os.path.join(folder, 'images/training')
            mask_folder = os.path.join(folder, 'annotations/training')
        else:
            img_folder = os.path.join(folder, 'images/validation')
            mask_folder = os.path.join(folder, 'annotations/validation')
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return np.array(img_paths), np.array(mask_paths)
    
    def _adapt_places(self, file):
        self.places_class_names = list()
        with open(file) as class_file: 
            for line in class_file:
                self.places_class_names.append(line.strip().split(" ")[0][3:])
        self.places_class_names = np.array(self.places_class_names)
        presence = np.isin(self.label_text_ADE, self.places_class_names)
        self.label_text_ADE = self.label_text_ADE[presence]
        self.images = self.images[presence]
        self.masks = self.masks[presence]
        outer = np.equal.outer(self.label_text_ADE, self.places_class_names)
        self.label_idx_PLACES = np.concatenate([np.argwhere(lbl == 1) for lbl in outer]).reshape(-1)
