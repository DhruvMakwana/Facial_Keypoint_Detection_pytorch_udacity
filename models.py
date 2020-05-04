## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 32, 5)     # (32, 220, 220) (224-5)/1 + 1
        
        # first pooling layer
        self.pool1 = nn.MaxPool2d(2, 2)      # (32, 110, 110)
        
        # second convolution layer
        self.conv2 = nn.Conv2d(32, 64, 5)    # (64, 106, 106) (110-5)/1 + 1
        
        # second pooling layer
        self.pool2 = nn.MaxPool2d(2, 2)      # (64, 53, 53)
        
        # third convolution layer
        self.conv3 = nn.Conv2d(64, 128, 5)    # (128, 49, 49) (53-5)/1 + 1
        
        # third pooling layer
        self.pool3 = nn.MaxPool2d(2, 2)      # (128, 24, 24)
        
        # dropout layer
        self.drop1 = nn.Dropout(p=0.25)
        
        # Fully connected layer
        
        self.fc1 = nn.Linear(128*24*24, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop1(x)
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
