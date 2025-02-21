import sys, os
import pickle
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, train=True, transform=None,rotations_transform = None):
        if train:
            data_filename = '/train.pkl'
        else:
            data_filename = '/test.pkl'
        file_path = data_path+data_filename

        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.images = data_dict[0]
        self.labels = data_dict[1].astype(np.int64).squeeze()
        self.transform = transform
        self.channels = channels
        self.rotations_transform = rotations_transform


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            if self.rotations_transform is not None:
                image = self.rotations_transform(torch.from_numpy(image).unsqueeze(0)).squeeze(0)
            image = Image.fromarray(np.array(image), mode='L')
        elif self.channels == 3:
            if self.rotations_transform is not None:
                image = self.rotations_transform(torch.from_numpy(image))
            image = Image.fromarray(np.array(image), mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AddSaltPepperNoise(object):
    def __init__(self, prob=0.05):
        self.prob = prob

    def __call__(self, tensor):
        noise_tensor = torch.rand(tensor.size())
        salt = noise_tensor < self.prob / 2
        pepper = noise_tensor > (1 - self.prob / 2)
        tensor[salt] = 1
        tensor[pepper] = 0
        return tensor

class R_MnistDataset(Dataset):
    def __init__(self, data_path, channels, train=True, rotations_transform=None,transform = None):
        if train:
            data_filename = '/train.pkl'
        else:
            data_filename = '/test.pkl'
        file_path = data_path+data_filename

        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.images = data_dict[0]
        self.labels = data_dict[1].astype(np.int64).squeeze()
        self.rotations_transform = rotations_transform
        self.channels = channels
        self.transform = transform


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            if self.rotations_transform is not None:
                image = self.rotations_transform(torch.from_numpy(image).unsqueeze(0)).squeeze(0)
            image = Image.fromarray(image.numpy(), mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        # Add different noise based on rotation angle
        # if self.rotations_transform is not None:
        #     rotation_angle = self.rotations_transform.degrees[0]  # Get the rotation angle
        #     if rotation_angle == 0:
        #         noise_transform = AddGaussianNoise(mean=0., std=0.1)
        #     elif rotation_angle == 90:
        #         noise_transform = AddSaltPepperNoise(prob=0.1)
        #     elif rotation_angle == 180:
        #         noise_transform = AddGaussianNoise(mean=0., std=0.2)
        #     elif rotation_angle == 270:
        #         noise_transform = AddSaltPepperNoise(prob=0.2)
        #     else:
        #         noise_transform = None
        #
        #     if noise_transform is not None:
        #         image = noise_transform(image)

        return image, label

class R_CifarDataset(Dataset):
    def __init__(self, data_path, channels, train=True, rotations_transform=None,transform = None):
        if train:
            data_filename = '/train.pkl'
        else:
            data_filename = '/test.pkl'
        file_path = data_path+data_filename

        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.images = data_dict[0]
        self.labels = data_dict[1].astype(np.int64).squeeze()
        self.rotations_transform = rotations_transform
        self.channels = channels
        self.transform = transform


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.channels == 1:
            if self.rotations_transform is not None:
                image = self.rotations_transform(torch.from_numpy(image).unsqueeze(0)).squeeze(0)
            image = Image.fromarray(image.numpy(), mode='L')
        elif self.channels == 3:
            if self.rotations_transform is not None:
                image = self.rotations_transform(torch.from_numpy(image))
            # image = Image.fromarray(image.numpy().transpose(1,2,0), mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label



def prepare_data(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset      = DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)

    # USPS
    usps_trainset      = DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset     = DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset     = DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders