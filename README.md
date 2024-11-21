Video Generation using Diffusion Transformers in PyTorch
========

## Building Video Generation Model Tutorial
<a href="https://www.youtube.com/watch?v=KAYYo3lNOHY">
   <img alt="Video generation with Diffusion Transformers" src="https://github.com/user-attachments/assets/7f987db0-26a1-490b-9ab0-60e26fa17d06"
   width="400">
</a>


## Sample Output for Latte on moving mnist easy videos 
Trained for 300 epochs

![mnist0-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/a71397ae-5848-439a-94f6-4a73bc35bd4e)
![mnist1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/5c535116-95b1-46e3-86ef-0cec4b1e56c2)
![mnist3-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/5c4dfb2b-82b1-4bba-ac03-023ddbcf58ab)
![mnist4-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/7d549275-b6bd-4af3-8607-f8214831eb18)

## Sample Output for Latte on UCF101 videos
Trained for 500 epochs(needs more training)

<img src="https://github.com/user-attachments/assets/f7cc10bc-5c3a-4bb6-881a-f9ff88e40f37" width="100">
<img src="https://github.com/user-attachments/assets/92d6ac8f-256b-4642-8785-8115e8b71949" width="100">
<img src="https://github.com/user-attachments/assets/00dffe19-0729-48fe-9a27-3e265313e5f8" width="100">
<img src="https://github.com/user-attachments/assets/0d243ed1-5c30-4965-854c-7d437f8b347a" width="100">



___  
This repository implements Latent Diffusion Transformer for Video Generation Paper. It provides code for the following:
* Training and inference of VAE on Moving Mnist and UCF101 frames
* Training and Inference of Latte Video Model using trained VAE on 16 frame video clips of both datasets
* Configurable code for training all models from Latte-S to Latte-XL

This repo has few changes from the  [official Latte implementation](https://github.com/Vchitect/Latte) except the following changes.
* Current code is for unconditional generation
* Variance is fixed during training and not learned (like original DDPM)
* No EMA 
* Ability to train VAE
* Ability to save latents of video frames for faster training


## Setup
* Create a new conda environment with python 3.10 then run below commands
* `conda activate <environment_name>`
* ```git clone https://github.com/explainingai-code/VideoGeneration-PyTorch.git```
* ```cd VideoGeneration-PyTorch```
* ```pip install -r requirements.txt```
*  Download lpips weights by opening this link in browser(dont use wget) https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth and downloading the raw file. Place the downloaded weights file in ```model/weights/v0.1/vgg.pth```
___  

## Data Preparation

### Moving Mnist Easy Videos
For moving mnist I have used the easy category of videos which have one number moving across frames.
Download the videos from [this](https://www.kaggle.com/datasets/yichengs/captioned-moving-mnist-dataset-easy-version/) page. This includes captions as well so one can also try with text to video models using this.
Create a `data` directory in the repo root and add the downloaded `mmnist-easy` folder there.
Ensure directory structure is the following
```
VideoGeneration-PyTorch
    -> data
        -> mmnist-easy
            -> *.mp4             
```

For setting up UCF101, simply download the videos from the official page [here](https://www.crcv.ucf.edu/data/UCF101.php)
and add it to `data` directory.
Ensure directory structure is the following
```
VideoGeneration-PyTorch
    -> data
        -> UCF101
            -> *.avi

```
---
## Configuration
 Allows you to play with different components of Latte and autoencoder
* ```config/mnist.yaml``` - Configuration used for moving mnist dataset
* ```config/ucf.yaml``` - Configuration used for ucf dataset
<ins>Important configuration parameters</ins>
* `autoencoder_acc_steps` : For accumulating gradients if video size is too large and a large batch size cant be used.

___  
## Training
The repo provides training and inference for Moving Mnist (Unconditional Latte Model)

For working on your own dataset:
* Create your own config and ensure following config parameters are correctly set 
  * `im_path` -  Folder name where latents will be saved (when `save_latents` script is run later)
  * `video_path`- Path to the videos
  * `video_ext` - Extension for videos. Assumption is all videos are same Extension
  * `frame_height`, `frame_width`, `frame_channels' - Dimension to which frames would be resized to.
  * `centre_square_crop` - If center cropping is needed on frames or not
  * `video_filter_path` - null / location of txt file which contains video names that need to be taken for training. If `null` then all videos will be used(like moving mnist). For seeing how to construct this filter file look at `dataset/ucf_filter.txt` file
* The existing `video_dataset.py` should require minimal modifications to adapt to your dataset requirements 

Once the config and dataset is setup:
* First train the auto encoder on your dataset using [this section](#training-autoencoder-for-latte)
* For training and inference of Unconditional Latte follow [this section](#training-unconditional-latte)

## Training AutoEncoder for Latte
* We need to first extract frames for training autoencoder
  *  By default, we extract only 10% of frames from our videos. Change `ae_frame_sample_prob` in `dataset_params` of config if you want to train on larger number of frames. For both moving mnist and ucf, 10% works fine.
  * If you need to train the latte model on a subset of videos then ensure the `video_filter_path` is correctly set. Look at ucf config and `dataset/ucf_filter.txt` for guidance.
  *  Extract the frames by running `python -m capture_frames --config config/mnist.yaml` with the right config value
* For training autoencoder 
  * Minimal modifications might be needed to `dataset/iamge_dataset.py` to adapt to your own dataset.
  * Make sure the `frame_height` and `frame_width` parameters are correctly set for resizing frames(if needed) 
  * Run ```python -m tools.train_vae --config config/mnist.yaml``` for training autoencoder with the right config file
* In case you gpu memory is limited, I would suggest to run `python -m tools.save_latents --config config/mnist.yaml` with the correct config. This script would save the latent frames for all your dataset videos. Otherwise during diffusion model training we would have to load vae also.

## Training Unconditional Latte
Train the autoencoder first and make changes to config and `video_dataset.py`(if any needed) to adapt to your requirements.

* ```python -m tools.train_vae_ditv --config config/mnist.yaml``` for training unconditional Latte using right config
* ```python -m tools.sample_vae_ditv --config config/mnist.yaml``` for generating videos using trained Latte model


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created.

During frame extraction , folder name for the key `im_path` will be created in `task_name` directory and frames will be saved in there.

During training of autoencoder the following output will be saved 
* Latest Autoencoder and discriminator checkpoint in ```task_name``` directory
* Sample reconstructions in ```task_name/vae_autoencoder_samples```

If `save_latents` script is run then latents will be saved in ```task_name/save_video_latent_dir``` if mentioned in config

During training and inference of unconditional Latte following output will be saved:
* During training we will save the latest checkpoint in ```task_name``` directory
* During sampling, unconditional sampled video generated for all timesteps will be saved in ```task_name/samples/*.mp4``` . The final decoded generated video will be `sample_video_0.mp4`. Videos from `sample_output_999.mp4` to `sample_output_1.mp4` will be latent video predictions of denoising process from T=999 to T=1. Final Generated Video is at T=0



## Citations
```
@misc{ma2024lattelatentdiffusiontransformer,
      title={Latte: Latent Diffusion Transformer for Video Generation}, 
      author={Xin Ma and Yaohui Wang and Gengyun Jia and Xinyuan Chen and Ziwei Liu and Yuan-Fang Li and Cunjian Chen and Yu Qiao},
      year={2024},
      eprint={2401.03048},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.03048}, 
}
```





