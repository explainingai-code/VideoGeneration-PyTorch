import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2 as v2
from model.vae import VAE
from model.transformer import DITVideo
from scheduler.linear_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, scheduler, train_config, ditv_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    latent_frame_height = (dataset_config['frame_height']
                           // 2 ** sum(autoencoder_model_config['down_sample']))
    latent_frame_width = (dataset_config['frame_width']
                          // 2 ** sum(autoencoder_model_config['down_sample']))

    xt = torch.randn((1, dataset_config['num_frames'],
                      autoencoder_model_config['z_channels'],
                      latent_frame_height, latent_frame_width)).to(device)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt,
                                                     noise_pred,
                                                     torch.as_tensor(i).to(device))

        # Save x0
        if i == 0:
            # Decode ONLY the final video to save time
            ims = vae.to(device).decode(xt[0])
        else:
            ims = xt

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        tv_frames = ims * 255

        if i == 0:
            tv_frames = tv_frames.permute((0, 2, 3, 1))
            if tv_frames.shape[-1] == 1:
                tv_frames = tv_frames.repeat((1, 1, 1, 3))
        else:
            tv_frames = v2.Compose([
                v2.Resize((dataset_config['frame_height'],
                           dataset_config['frame_width']),
                          interpolation=v2.InterpolationMode.NEAREST),
            ])(tv_frames)
            tv_frames = tv_frames[0].permute((0, 2, 3, 1))[:, :, :, :3]

        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        torchvision.io.write_video(os.path.join(train_config['task_name'],
                                                'samples/sample_output_{}.mp4'.format(i)),
                                   tv_frames,
                                   fps=8)


def infer(args):
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

    # Get latent image size
    frame_height = (dataset_config['frame_height']
                    // 2 ** sum(autoencoder_model_config['down_sample']))
    frame_width = (dataset_config['frame_width']
                   // 2 ** sum(autoencoder_model_config['down_sample']))
    num_frames = dataset_config['num_frames']
    model = DITVideo(frame_height=frame_height,
                     frame_width=frame_width,
                     im_channels=autoencoder_model_config['z_channels'],
                     num_frames=num_frames,
                     config=ditv_model_config).to(device)
    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['ditv_ckpt_name'])), \
        "Train DiT Video Model first"

    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ditv_ckpt_name']),
                                     map_location=device))
    print('Loaded dit video checkpoint')

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    vae = VAE(im_channels=dataset_config['frame_channels'],
              model_config=autoencoder_model_config)
    vae.eval()

    # Load vae if found
    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vae_autoencoder_ckpt_name'])), \
        "VAE checkpoint not present. Train VAE first."
    vae.load_state_dict(torch.load(
        os.path.join(train_config['task_name'],
                     train_config['vae_autoencoder_ckpt_name']),
        map_location=device), strict=True)
    print('Loaded vae checkpoint')

    with torch.no_grad():
        sample(model, scheduler, train_config, ditv_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for latte video generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
