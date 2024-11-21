import argparse
import glob
import os
import pickle

import torch
import torchvision
import yaml
from dataset.ucf_dataset import UCFDataset
from dataset.mnist_dataset import MnistDataset
from dataset.video_dataset import VideoDataset

from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.v2
from torchvision.utils import make_grid
from tqdm import tqdm


from model.vae import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def save_vae_latents(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    model = VAE(im_channels=dataset_config['frame_channels'],
                model_config=autoencoder_config).to(device)
    model.load_state_dict(torch.load(os.path.join(
        train_config['task_name'],
        train_config['vae_autoencoder_ckpt_name']),
        map_location=device))
    model.eval()

    dataset = VideoDataset('train',
                           dataset_config)

    print('Will be generating latents for {} videos'.format(len(dataset.video_paths)))
    with torch.no_grad():
        if not os.path.exists(os.path.join(train_config['task_name'],
                                           train_config['save_video_latent_dir'])):
            os.mkdir(os.path.join(train_config['task_name'],
                                  train_config['save_video_latent_dir']))
        for path in tqdm(dataset.video_paths):
            # Read the video
            frames, _, _ = torchvision.io.read_video(filename=path,
                                                     pts_unit='sec',
                                                     output_format='TCHW')

            # Transform all frames
            frames_tensor = dataset.transforms(frames)
            if dataset_config['frame_channels'] == 1:
                frames_tensor = frames_tensor[:, 0:1, :, :]

            encoded_outputs = []
            for frame_tensor in frames_tensor:
                _, encoded_output = model.encode(
                    frame_tensor.float().unsqueeze(0).to(device)
                )
                encoded_outputs.append(encoded_output)
            encoded_outputs = torch.cat(encoded_outputs, dim=0)
            pickle.dump(encoded_outputs, open(
                os.path.join(train_config['task_name'],
                             train_config['save_video_latent_dir'],
                             '{}.pkl'.format(os.path.basename(path))), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vae inference and '
                                                 'saving latents')
    parser.add_argument('--config', dest='config_path',
                        default='config/ucf.yaml', type=str)
    args = parser.parse_args()
    save_vae_latents(args)
