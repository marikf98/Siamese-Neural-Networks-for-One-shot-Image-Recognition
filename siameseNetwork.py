import torch
import torch.nn as nn
import torch.nn.functional as F


# Input: two images
# Output: a similarity score between 0 and 1 - probability of being the same class

class SiameseNetwork(
    nn.Module):  # nn.Module is the base class for all neural network modules in PyTorch we inherite from it so we can get all the base functionality
    def __init__(self):
        super(SiameseNetwork, self).__init__()  # call the constructor of the parent class - nn.Module
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=10)  # Extracts 64 feature maps from the 1-channel grayscale input using 10×10 filters
        self.relu1 = nn.ReLU()  # Applies non-linearity
        self.pool1 = nn.MaxPool2d(
            kernel_size=2)  # Downsamples the output by a factor of 2 - each feature map was downsampled to half its size 96*96 - 48*48

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=7)  # Extracts 128 feature maps from the 64-channel input using 7×7 filters
        self.relu2 = nn.ReLU()  # Applies non-linearity
        self.pool2 = nn.MaxPool2d(
            kernel_size=2)  # Downsamples the output by a factor of 2 - each feature map was downsampled to half its size 42*42 - 21*21

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=4)  # Extracts 128 feature maps from the 128-channel input using 4×4 filters
        self.relu3 = nn.ReLU()  # Applies non-linearity
        self.pool3 = nn.MaxPool2d(
            kernel_size=2)  # Downsamples the output by a factor of 2 - each feature map was downsampled to half its size 18*18 - 9*9

        # Forth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=4)  # Extracts 256 feature maps from the 128-channel input using 4×4 filters
        self.relu4 = nn.ReLU()  # Applies non-linearity
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=9216,
                             out_features=4096)  # Fully connected layer - 9216 is the number of features after flattening the output of the last convolutional layer 256*6*6
        self.sigmoid = nn.Sigmoid()  # Applies non-linearity

        self.out = nn.Linear(in_features=4096, out_features=1)

    def forward(self, x1, x2):  # x1, x2 are the two input images
        # Encode the first image
        out1 = self.conv1(x1)  # Apply the first convolutional layer to the first image
        out1 = self.relu1(out1)  # Apply the ReLU activation function
        out1 = self.pool1(out1)  # Apply the first max pooling layer to the first image

        out1 = self.conv2(out1)  # Apply the second convolutional layer to the first image
        out1 = self.relu2(out1)  # Apply the ReLU activation function
        out1 = self.pool2(out1)  # Apply the second max pooling layer to the first image

        out1 = self.conv3(out1)  # Apply the third convolutional layer to the first image
        out1 = self.relu3(out1)  # Apply the ReLU activation function
        out1 = self.pool3(out1)  # Apply the third max pooling layer to the first image

        out1 = self.conv4(out1)  # Apply the forth convolutional layer to the first image
        out1 = self.relu4(out1)  # Apply the ReLU activation function

        out1 = self.flatten(out1)  # Flatten the output of the last convolutional layer
        out1 = self.fc1(out1)  # Apply the fully connected layer to the first image
        out1 = self.sigmoid(out1)  # Apply the sigmoid activation function to the first image to scale the vector value to [0,1]

        # Encode the second image
        out2 = self.conv1(x2) # Apply the first convolutional layer to the second image
        out2 = self.relu1(out2) # Apply the ReLU activation function
        out2 = self.pool1(out2) # Apply the first max pooling layer to the second image

        out2 = self.conv2(out2) # Apply the second convolutional layer to the second image
        out2 = self.relu2(out2) # Apply the ReLU activation function
        out2 = self.pool2(out2) # Apply the second max pooling layer to the second image

        out2 = self.conv3(out2) # Apply the third convolutional layer to the second image
        out2 = self.relu3(out2) # Apply the ReLU activation function
        out2 = self.pool3(out2) # Apply the third max pooling layer to the second image

        out2 = self.conv4(out2) # Apply the forth convolutional layer to the second image
        out2 = self.relu4(out2) # Apply the ReLU activation function

        out2 = self.flatten(out2) # Flatten the output of the last convolutional layer
        out2 = self.fc1(out2) # Apply the fully connected layer to the second image
        out2 = self.sigmoid(out2) # Apply the sigmoid activation function to the second image to scale the vector value to [0,1]

        distance = torch.abs(out1 - out2)  # Compute the absolute difference between the two outputs
        score = self.out(distance)  # Apply the final fully connected layer to the distance

        return score