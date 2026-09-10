from os.path import join
from PIL import Image
import datasets.transforms_video as TV
from torch.utils.data import Dataset
import torchvision.transforms as TF
import random
import numpy as np
import torch
from tools.colormap import colormap
import os

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.25 + color * 0.75
    origin_img = Image.fromarray(origin_img)
    return origin_img


def denormalize(tens):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    return (tens*std)+mean


def make_video_transforms(image_set, max_size=1024, resize=False, hflip=False):
    normalize = TV.Compose([
        TV.ToTensor(),
        TV.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [320, 392, 448, 480, 512, 640, 720]
    if not resize or (image_set != 'train'):
        transforms = []
        if image_set == 'train' and hflip:
            transforms.append(TV.RandomHorizontalFlip())
        transforms.extend([
            TV.RandomResize([max_size], max_size=max_size),
            normalize,
        ])
        return TV.Compose(
            transforms
        )
    if image_set == 'train':
        transforms = [TV.PhotometricDistort()]
        if hflip:
            transforms.append(TV.RandomHorizontalFlip())
        transforms.extend([
            TV.Compose([
                TV.RandomResize(scales, max_size=max_size),
                TV.Check(),
            ]),
            normalize,
        ])
        return TV.Compose(transforms)
    
    raise ValueError(f'unknown {image_set}')


class FrameSampler:
    @staticmethod
    def sample_local_frames(
        frame_id, vid_len, num_frames, temporal_strides=(1, 2)
    ):
        """Sample an ordered constant-stride clip containing the anchor frame."""
        if vid_len < 1 or num_frames < 1:
            raise ValueError("vid_len and num_frames must be positive")
        if not 0 <= frame_id < vid_len:
            raise ValueError("frame_id must be inside the video")

        strides = sorted({int(stride) for stride in temporal_strides})
        if not strides or strides[0] < 1:
            raise ValueError("temporal strides must be positive")

        if vid_len < num_frames:
            if num_frames == 1:
                return [frame_id]
            return [
                round(index * (vid_len - 1) / (num_frames - 1))
                for index in range(num_frames)
            ]

        valid_strides = [
            stride
            for stride in strides
            if (num_frames - 1) * stride < vid_len
        ]
        stride = random.choice(valid_strides or [1])

        min_anchor_slot = max(
            0,
            (
                frame_id
                + (num_frames - 1) * stride
                - (vid_len - 1)
                + stride
                - 1
            )
            // stride,
        )
        max_anchor_slot = min(num_frames - 1, frame_id // stride)
        anchor_slot = random.randint(min_anchor_slot, max_anchor_slot)
        start = frame_id - anchor_slot * stride
        return [start + index * stride for index in range(num_frames)]

    @staticmethod
    def sample_from_pool(frame_pool, sample_indx, num_frames):
        iters = 0
        max_iter = len(frame_pool)*3
        while (len(sample_indx) < num_frames) and (iters < max_iter):
            samp_frame = random.sample(frame_pool, 1)[0]
            if samp_frame not in sample_indx:
                sample_indx.append(samp_frame)
            iters += 1
        
        return len(sample_indx) == num_frames

    @staticmethod
    def sample_global_frames(frame_id, vid_len, num_frames):
        # random sparse sample
        sample_indx = [frame_id]
        if num_frames != 1:
            # local sample
            sample_id_before = random.randint(1, 3)
            sample_id_after = random.randint(1, 3)
            local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >= global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    while len(sample_indx) < num_frames:
                        samp_frame = random.sample(range(vid_len), 1)[0]
                        sample_indx.append(samp_frame)

        sample_indx.sort()
        return sample_indx


class VideoEvalDataset(Dataset):
    def __init__(self, vid_folder, frames, ext='.jpg', max_size=1024):
        super().__init__()
        self.vid_folder = vid_folder
        self.frames = frames
        self.vid_len = len(frames)
        self.ext = ext
        with Image.open(join(vid_folder, frames[0] + ext)) as image:
            self.origin_w, self.origin_h = image.size
        self.transform = TF.Compose([
            TF.Resize(max_size-4, max_size=max_size), #T.Resize(360),
            TF.ToTensor(),
            TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
    def __len__(self):
        return self.vid_len
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        img_path = join(self.vid_folder, frame + self.ext)
        img = Image.open(img_path).convert('RGB')
        
        return self.transform(img), idx
