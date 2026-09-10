"""
DTA-SAM infrared video data loader
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

from datasets.categories import ytvos_category_dict as category_dict
from datasets.transform_utils import make_video_transforms, FrameSampler


class InfraredVideoDataset(Dataset):
    """Text-free infrared clips in the shared sequence/annotation layout."""
    def __init__(self, img_folder: Path, ann_file: Path, transforms,
                 num_frames: int, temporal_strides=(1,), native_mask_loss=False):
        self.img_folder = img_folder     
        self.ann_file = ann_file         
        self._transforms = transforms    
        self.num_frames = num_frames
        self.temporal_strides = tuple(temporal_strides)
        self.native_mask_loss = native_mask_loss
        # create video meta data
        self.prepare_metas()

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())
        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            object_ids = {
                str(exp_dict['obj_id'])
                for exp_dict in vid_data['expressions'].values()
            }
            for obj_id in sorted(object_ids, key=int):
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['obj_id'] = int(obj_id)
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # get positives and negatives
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)

        print("Total clips: ", len(self.metas))

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, obj_id, category, frames, frame_id = \
                        meta['video'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
            category_id = category_dict[category]
            vid_len = len(frames)

            sample_indx = FrameSampler.sample_local_frames(
                frame_id,
                vid_len,
                self.num_frames,
                self.temporal_strides,
            )

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')

                # create the target
                label =  torch.tensor(category_id) 
                mask = np.array(mask)
                mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0) 
            boxes = torch.stack(boxes, dim=0) 
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)
            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'labels': labels,                        # [T,]
                'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'orig_size': torch.as_tensor([int(h), int(w)]), 
                'size': torch.as_tensor([int(h), int(w)]),
            }
            if self.native_mask_loss:
                target['native_masks'] = masks.clone()

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target) 
            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
            
            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target



def build(image_set, args):
    root = Path(args.data_root)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    transforms = make_video_transforms(
        image_set,
        max_size=args.max_size,
        resize=False,
        hflip=getattr(args, "augm_hflip", False),
    )
    dataset = InfraredVideoDataset(
        img_folder,
        ann_file,
        transforms=transforms,
        num_frames=args.num_frames,
        temporal_strides=(1,),
        native_mask_loss=getattr(args, "native_mask_loss", False),
    )
    return dataset
