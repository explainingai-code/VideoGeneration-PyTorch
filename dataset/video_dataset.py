import glob
import os
import cv2
import torch
import numpy as np
import random
import torchvision
import pickle
import torchvision.transforms.v2
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class VideoDataset(Dataset):
    r"""
    Simple Video Dataset class for training diffusion transformer
    for video generation.
    If latents are present, the dataset uses the saved latents for the videos,
    else it reads the video and extracts frames from it.
    """
    def __init__(self, split, dataset_config, latent_path=None, im_ext='png'):
        r"""
        Initialize all parameters and also check
        if latents are present or not
        :param split: for now this is always train
        :param dataset_config: config parameters for dataset(mnist/ucf)
        :param latent_path: Path for saved latents
        :param im_ext: assumes all images are of this extension. Used only
        if latents are not present
        """
        self.split = split
        self.video_ext = dataset_config['video_ext']
        self.num_images = dataset_config['num_images_train']
        self.use_images = self.num_images > 0
        self.num_frames = dataset_config['num_frames']
        self.frame_interval = dataset_config['frame_interval']
        self.frame_height = dataset_config['frame_height']
        self.frame_width = dataset_config['frame_width']
        self.frame_channels = dataset_config['frame_channels']
        self.center_square_crop = dataset_config['centre_square_crop']
        self.filter_fpath = dataset_config['video_filter_path']
        if self.center_square_crop:
            assert self.frame_height == self.frame_width, \
                ('For centre square crop frame_height '
                 'and frame_width should be same')
            self.transforms = torchvision.transforms.v2.Compose([
                    torchvision.transforms.v2.Resize(self.frame_height),
                    torchvision.transforms.v2.CenterCrop(self.frame_height),
                    torchvision.transforms.v2.ToPureTensor(),
                    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                    torchvision.transforms.v2.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transforms = torchvision.transforms.v2.Compose([
                    torchvision.transforms.v2.Resize((self.frame_height,
                                                      self.frame_width)),
                    torchvision.transforms.v2.ToPureTensor(),
                    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                    torchvision.transforms.v2.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                ])

        # Load video paths for this dataset
        self.video_paths = self.load_videos(dataset_config['video_path'],
                                            self.filter_fpath)

        # Validate if latents are present and if they are present for all
        # videos. And only upon validation set `use_latents` as True
        self.use_latents = False
        if latent_path is not None and os.path.exists(latent_path):
            num_latents = len(glob.glob(os.path.join(latent_path, '*.pkl')))
            self.latents = glob.glob(os.path.join(latent_path, '*.pkl'))
            if num_latents == len(self.video_paths):
                self.use_latents = True
                print('Will use latents')

    def load_videos(self, video_path, filter_fpath=None):
        r"""
        Method to load all video names for training.
        This uses the filter file to use only selective videos
        for training.
        :param video_path: Path for all videos in the dataset
        :param filter_fpath: Path for file containing filters for relevant videos
        :return:
        """
        assert os.path.exists(video_path), (
            "video path {} does not exist".format(video_path))
        video_paths = []

        if filter_fpath is None:
            filters = ['*']
        else:
            filters = []
            assert os.path.exists(filter_fpath), "Filter file not present"
            with open(filter_fpath, 'r') as f:
                for line in f.readlines():
                    filters.append(line.strip())
        for filter in filters:
            for fname in glob.glob(os.path.join(video_path,
                                                '{}.{}'.format(filter,
                                                               self.video_ext))):
                video_paths.append(fname)
        print('Found {} videos'.format(len(video_paths)))
        return video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        # We do things differently whether we are working with latents or not
        if self.use_latents:
            # Load the latent corresponding to this item
            latent_path = self.latents[index]
            latent = pickle.load(open(latent_path, 'rb')).cpu()

            # Sample (self.frame_interval * self.num_frames) frames
            # and from that take num_frames(16) equally spaced frames
            # Keep only the latents of these sampled frames
            num_frames = len(latent)
            total_frames = self.frame_interval * self.num_frames
            max_end = max(0, num_frames - total_frames - 1)
            start_index = random.randint(0, max_end)
            end_index = min(start_index + total_frames, num_frames)
            frame_idxs = np.linspace(start_index, end_index - 1, self.num_frames,
                                     dtype=int)
            latent = latent[frame_idxs]

            # From the latent extract the mean and std
            # and reparametrization to sample according to this
            # mean and std
            mean, logvar = torch.chunk(latent, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            frames_tensor = mean + std * torch.randn(mean.shape)

            # If we are doing image+video joint training
            # then sample random num_images(8) videos
            # and then sample a random latent frame from that video
            if self.use_images:
                im_latents = []
                for _ in range(self.num_images):
                    # Sample a random video
                    # From this video sample a random saved latent
                    # Extract mean, std from this latent and then
                    # use that to get a im_latent
                    video_idx = random.randint(0, len(self.latents)-1)
                    random_video_latent = pickle.load(open(self.latents[video_idx],
                                                           'rb')).cpu()
                    frame_idx = random.randint(0, len(random_video_latent) - 1)
                    im_latent = random_video_latent[frame_idx]
                    mean, logvar = torch.chunk(im_latent, 2, dim=0)
                    std = torch.exp(0.5 * logvar)
                    im_latent = mean + std * torch.randn(mean.shape)
                    im_latents.append(im_latent.unsqueeze(0))

                im_latents = torch.cat(im_latents)

                # Concat video latents and image latents together for training
                frames_tensor = torch.cat([frames_tensor, im_latents], dim=0)
            return frames_tensor
        else:
            # Read the video corresponding to this item
            path = self.video_paths[index]
            frames, _, _ = torchvision.io.read_video(filename=path,
                                                     pts_unit='sec',
                                                     output_format='TCHW')

            # Sample (self.frame_interval * self.num_frames) frames
            # and from that take num_frames(16) equally spaced frames
            # Keep only these sampled frames
            num_frames = len(frames)
            max_end = max(0, num_frames - (self.num_frames * self.frame_interval) - 1)
            start_index = random.randint(0, max_end)
            end_index = min(start_index + (self.num_frames * self.frame_interval),
                            num_frames)
            frame_idxs = np.linspace(start_index, end_index - 1, self.num_frames,
                                     dtype=int)
            frames = frames[frame_idxs]

            # Resize frames according to transformations
            # desired based on config
            frames_tensor = self.transforms(frames)

            # For grayscale keep only the first channel
            if self.frame_channels == 1:
                frames_tensor = frames_tensor[:, 0:1, :, :]

            # If we are doing image+video joint training
            # then sample random num_images(8) videos
            # and then sample a random frame from that video
            if self.use_images:
                im_tensors = []
                for _ in range(self.num_images):
                    # Sample a random video
                    # From this video sample a random image frame
                    video_idx = random.randint(0, len(self.video_paths)-1)
                    path = self.video_paths[video_idx]
                    ims, _, _ = torchvision.io.read_video(filename=path,
                                                          pts_unit='sec',
                                                          output_format='TCHW')
                    im_idx = random.randint(0, len(ims) - 1)
                    ims = ims[im_idx]

                    # Resize this sampled image according to transformations
                    # desired based on config
                    im_tensor = self.transforms(ims)

                    # For grayscale keep only the first channel
                    if self.frame_channels == 1:
                        im_tensor = im_tensor[0:1, :, :]
                    im_tensors.append(im_tensor.unsqueeze(0))

                im_tensors = torch.cat(im_tensors)
                frames_tensor = torch.cat([frames_tensor, im_tensors], dim=0)
            return frames_tensor
