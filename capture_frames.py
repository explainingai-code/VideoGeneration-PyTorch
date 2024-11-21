import cv2
import random
import argparse
import glob
import yaml
import os
from tqdm import tqdm


def extract_frames_from_video(video_path, im_path, frame_sample_prob, count):
    video_obj = cv2.VideoCapture(video_path)

    success = 1
    while success:
        success, image = video_obj.read()
        if not success:
            break
        if random.random() > frame_sample_prob:
            continue

        cv2.imwrite('{}/frame_{}.png'.format(im_path, count), image)
        count += 1
    return count


def extract_frames(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    task_name = config['train_params']['task_name']
    video_path = dataset_config['video_path']
    assert os.path.exists(video_path), "video path {} does not exist".format(video_path)

    im_path = os.path.join(task_name, dataset_config['im_path'])
    if not os.path.exists(im_path):
        os.mkdir(im_path)
    filter_fpath = dataset_config['video_filter_path']
    video_ext = dataset_config['video_ext']
    frame_sample_prob = dataset_config['ae_frame_sample_prob']

    # Create frames directory if not present
    if not os.path.exists(im_path):
        os.mkdir(im_path)

    video_paths = []
    if filter_fpath is None:
        filters = ['*']
    else:
        filters = []
        assert os.path.exists(filter_fpath), "Filter file not present"
        with open(filter_fpath, 'r') as f:
            for line in f.readlines():
                filters.append(line.strip())
    for filter_i in filters:
        for fname in glob.glob(os.path.join(video_path, '{}.{}'.format(filter_i,
                                                                       video_ext))):
            video_paths.append(fname)

    print('Found {} videos'.format(len(video_paths)))
    print('Extracting frames....')
    count = 0
    for video_path in tqdm(video_paths):
        count = extract_frames_from_video(video_path, im_path, frame_sample_prob, count)

    print('Extracted total {} frames'.format(count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for frame extraction '
                                                 'for autoencoder training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    extract_frames(args)
