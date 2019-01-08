import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from PIL import Image

import time
import argparse

from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training process")
    parser.add_argument('--data_dir', action='store', default='flowers')
    parser.add_argument('--arch', dest='arch', default='vgg13', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='8')
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()


def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    running_loss = 0
    accuracy = 0
    start = time.time()

    for e in range(epochs):
        
        for mode in ['train', 'validate']:   
            if mode == 'train':
                model.train()
            else:
                model.eval()
            
            pass_count = 0       
            for data in dataloaders[mode]:
                pass_count += 1
                inputs, labels = data
                if gpu and cuda:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                # Forward
                outputs = model.forward(inputs)
                # _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # Backward
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item()
                ps = torch.exp(outputs).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
                #ps = torch.exp(outputs).data
                #equality = (labels.data == ps.max(1)[1])
                #if gpu and cuda:
                 #   accuracy += equality.type_as(torch.cuda.FloatTensor()).mean()
                #else:
                 #   accuracy += equality.type_as(torch.FloatTensor()).mean()
            
            if mode == 'train':
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count),
                      "Training Accuracy: {:.4f}".format(accuracy))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                  "Validation Accuracy: {:.4f}".format(accuracy))
            running_loss = 0

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            

def main():
    args = parse_args()
    
    #data_dir = 'flowers'
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    validataion_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])])
    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 
    image_datasets = dict()
    image_datasets['train'] = ImageFolder(train_dir, transform=training_transforms)
    image_datasets['validate'] = ImageFolder(valid_dir, transform=validataion_transforms)
    image_datasets['test'] = ImageFolder(test_dir, transform=testing_transforms)
    
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    dataloaders['validate'] = torch.utils.data.DataLoader(image_datasets['validate'], batch_size=64)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        
    elif args.arch == "densenet121":
        feature_num = model.classifier.in_features
    
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(feature_num, int(args.hidden_units))),
                                ('drop', nn.Dropout(p=0.5)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(int(args.hidden_units), 102)),
                                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = image_datasets['train'].class_to_idx
    gpu = args.gpu
    train(model, criterion, optimizer, dataloaders ,epochs, gpu)
    model.class_to_idx = class_index
    save_checkpoint(model, optimizer, args, classifier)


if __name__ == "__main__":
    main()