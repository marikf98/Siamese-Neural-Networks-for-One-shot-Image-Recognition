import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    """
    Siamese Network for one-shot image recognition, based on the architecture
    from the paper "Siamese Neural Networks for One-shot Image Recognition".

    The network takes two grayscale images as input, processes them through
    shared convolutional layers, computes a 4096-dimensional embedding for
    each, calculates the L1 distance between embeddings, and outputs a
    similarity score between 0 and 1.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            # Conv Layer 1: Input 1x105x105 -> Output 64x96x96
            nn.Conv2d(1, 64, kernel_size=10),  # -> 96x96
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 48x48

            # Conv Layer 2: -> 128x42x42
            nn.Conv2d(64, 128, kernel_size=7),# -> 42x42
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 21x21

            # Conv Layer 3: -> 128x18x18
            nn.Conv2d(128, 128, kernel_size=4),# -> 18x18
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 9x9
            
            # Conv Layer 4: -> 256x6x6
            nn.Conv2d(128, 256, kernel_size=4),# -> 6x6
            nn.ReLU()
        )

        # Flatten the output of the CNN: 256 * 6 * 6 = 9216
        self.flatten = nn.Flatten()
        # Fully connected layer to create the feature vector (4096-dim) with sigmoid activation
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid()  
        )
        # Final layer to map the L1 distance vector to a similarity score
        self.out = nn.Linear(4096, 1)

    def forward_once(self, x):
        """
       Computes a feature embedding for a single input image.

       Args:
           x (torch.Tensor): A batch of grayscale images of shape (batch_size, 1, 105, 105).

       Returns:
           torch.Tensor: A tensor of shape (batch_size, 4096) representing the image embeddings.
        """
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def forward(self, x1, x2):
        """
        Computes a similarity score between two input images.

        Args:
            x1 (torch.Tensor): A batch of first images with shape (batch_size, 1, 105, 105).
            x2 (torch.Tensor): A batch of second images with the same shape.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) with similarity scores in [0, 1],
                          where 1 indicates high similarity (same class) and 0 indicates low similarity.
        """
        # Encode both images using the shared CNN
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        # L1 distance between feature vectors
        distance = torch.abs(out1 - out2)
        # Pass through final layer
        score = self.out(distance)
        return score # The final sigmoid is applied internally in the loss function BCEWithLogitsLoss