import glob
import os
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset


class ImageDataset(Dataset):
    r"""
    Simple Image Dataset class for training autoencoder on frames
    of the videos
    """
    def __init__(self, split, dataset_config, task_name, im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param dataset_config: config parameters for dataset(mnist/ucf)
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_ext = im_ext
        self.frame_height = dataset_config['frame_height']
        self.frame_width = dataset_config['frame_width']
        self.frame_channels = dataset_config['frame_channels']
        self.center_square_crop = dataset_config['centre_square_crop']
        self.images = self.load_images(os.path.join(task_name, dataset_config['im_path']))

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        for fname in glob.glob(os.path.join(im_path, '*.{}'.format(self.im_ext))):
            ims.append(fname)
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        assert self.frame_channels in (1, 3), "Frame channels can only be 1/3"
        if self.frame_channels == 1:
            im = Image.open(self.images[index]).convert('L')
        else:
            im = Image.open(self.images[index]).convert('RGB')
        if self.center_square_crop:
            assert self.frame_height == self.frame_width, \
                ('For centre square crop frame_height '
                 'and frame_width should be same')
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.frame_height),
                torchvision.transforms.CenterCrop(self.frame_height),
                torchvision.transforms.ToTensor(),
            ])(im)
        else:
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.frame_height, self.frame_width)),
                torchvision.transforms.ToTensor(),
            ])(im)

        im.close()
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
