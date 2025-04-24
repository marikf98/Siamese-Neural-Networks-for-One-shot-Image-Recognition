import torch
from torch.utils.data import DataLoader

import siameseNetwork

from data_loader import data_loader as SiameseDataset
device = torch.device("cpu")
model = siameseNetwork.SiameseNetwork().to(device) # created an instance of the model and moved it to the device
loss_func = torch.nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-4)
data_set = SiameseDataset('./lfw2') # created an instance of the data loader

train_loader = DataLoader(data_set, batch_size=16, shuffle=True)

num_of_batches = len(train_loader)
num_of_epoch = 200
for epoch in range(num_of_epoch):
    model.train() # set the model to training mode
    running_loss = 0.0
    for img1, img2, label in train_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device).float()
        optimizer.zero_grad() # reset gradient from the previous iteration
        output = model(img1, img2).squeeze() # forward pass - is the same as to write model.forward(img1, img2) because we inherite from NN model will call forward
        loss = loss_func(output.view(-1), label.view(-1))
        loss.backward() # backpropagate so gradients are calculated.
        optimizer.step()
        running_loss += loss.item() # .item() to convert tensor to a float

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_of_epoch}], Loss: {avg_loss:.4f}")
