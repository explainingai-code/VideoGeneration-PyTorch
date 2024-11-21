import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset.video_dataset import VideoDataset
from model.transformer import DITVideo
from model.vae import VAE
from scheduler.linear_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    ditv_model_config = config['ditv_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    dataset = VideoDataset('train',
                           dataset_config=dataset_config,
                           latent_path=os.path.join(
                               train_config['task_name'],
                               train_config['save_video_latent_dir']))

    data_loader = DataLoader(dataset,
                             batch_size=train_config['ditv_batch_size'],
                             shuffle=True)

    # Instantiate the model
    frame_height = (dataset_config['frame_height'] //
                    2 ** sum(autoencoder_model_config['down_sample']))
    frame_width = (dataset_config['frame_width'] //
                   2 ** sum(autoencoder_model_config['down_sample']))
    num_frames = dataset_config['num_frames']
    model = DITVideo(frame_height=frame_height,
                     frame_width=frame_width,
                     im_channels=autoencoder_model_config['z_channels'],
                     num_frames=num_frames,
                     config=ditv_model_config).to(device)
    model.train()

    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ditv_ckpt_name'])):
        print('Loaded DiT Video checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ditv_ckpt_name']),
                                         map_location=device))

    # Load VAE
    if not dataset.use_latents:
        print('Loading vae model as latents not present')
        vae = VAE(im_channels=dataset_config['frame_channels'],
                  model_config=autoencoder_model_config).to(device)
        vae.eval()
        # Load vae if found
        assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vae_autoencoder_ckpt_name'])), \
            "VAE checkpoint not found"
        vae.load_state_dict(torch.load(os.path.join(
            train_config['task_name'],
            train_config['vae_autoencoder_ckpt_name']),
            map_location=device))
        print('Loaded vae checkpoint')
        for param in vae.parameters():
            param.requires_grad = False

    # Specify training parameters
    num_epochs = train_config['ditv_epochs']
    optimizer = AdamW(model.parameters(), lr=1E-4, weight_decay=0)
    criterion = torch.nn.MSELoss()

    acc_steps = train_config['ditv_acc_steps']
    for epoch_idx in range(num_epochs):
        losses = []
        step_count = 0
        for ims in tqdm(data_loader):
            step_count += 1
            optimizer.zero_grad()
            ims = ims.float().to(device)
            B, F, C, H, W = ims.shape
            if not dataset.use_latents:
                with torch.no_grad():
                    ims, _ = vae.encode(ims.reshape(-1, C, H, W))
                    ims = rearrange(ims, '(b f) c h w -> b f c h w', b=B, f=F)

            # Sample random noise
            noise = torch.randn_like(ims).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'],
                              (ims.shape[0],)).to(device)

            # Add noise to video according to timestep
            noisy_im = scheduler.add_noise(ims, noise, t)
            pred = model(noisy_im, t, num_images=dataset_config['num_images_train'])
            loss = criterion(pred, noise)
            losses.append(loss.item())
            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ditv_ckpt_name']))

    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for latte training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)
