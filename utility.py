import argparse
import torch
from torchvision import datasets, transforms, models
import json
from PIL import Image
import numpy as np

def get_input_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type = str, help='Folder name for image files.', default='flowers')
    parser.add_argument('--save_dir', help='Set directory to save checkpoints.', default='.')
    parser.add_argument('--arch',type = str, choices=['resnet18', 'vgg16', 'densenet121'], default='vgg16',help = 'the CNN model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default='2048', help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Set epochs')
    parser.add_argument('--gpu',  type=bool, default=False, const=True, nargs='?', help='train on gpu')

    return parser.parse_args()

def get_input_args_predict():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_image', help='Path to image')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Number of most likely classes')
    parser.add_argument('--category_names', default='./cat_to_name.json', help='Path of Category Mapping')
    parser.add_argument('--gpu', type=bool, default=False, const=True, nargs='?', help='use gpu')
    
    return parser.parse_args()

def set_device_type(gpu_mode):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu_mode) else "cpu")
    print("Device is set:",device)
    return device

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    #for validation and test
    test_n_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                                       [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_n_valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_n_valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    
    image_datasets = {
        'train' :train_data,
        'valid': valid_data,
        'test': test_data
    }

    dataloaders = {
        'train' : trainloader,
        'valid' : validloader,
        'test' : testloader
    }
    print("Data is loaded")
    return image_datasets, dataloaders

def load_category_map(cat_to_name='cat_to_name.json'):
    with open(cat_to_name, 'r') as f:
        category_names = json.load(f)
    print('Category map is loaded')    
    return category_names

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(244),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    image_pil = Image.open(image_path)
    image_pil = pil_transforms(image_pil)
    np_image = np.array(image_pil)
    return np_image