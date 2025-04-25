import torch
from torch.utils.data import DataLoader
import time
import copy
from data_loader import data_loader as SiameseDataset
from siameseNetwork import SiameseNetwork
import torch.nn as nn

def train_siamese_model(
    data_dir: str,
    pairs_file: str,
    device='cpu',
    batch_size=16,
    lr=0.01,
    weight_decay=0,
    epochs=50,
    print_every=1,
    augment_train=True,
    patience=10,
    return_best_model=True,
):
    """
     Trains a SiameseNetwork model with early stopping, optional augmentation,
        and logs training/validation loss per epoch.
    Args:
        data_dir (str): Root directory containing class-labeled subfolders.
        pairs_file (str): Path to .txt file defining image pairs.
        device (str): Device to train on ('cuda' or 'cpu').
        batch_size (int): Batch size for training and validation.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization factor.
        epochs (int): Maximum number of epochs.
        print_every (int): Frequency of printing loss information.
        augment_train (bool): Whether to apply data augmentation to training data.
        patience (int): Number of epochs to wait for improvement before early stopping.
        return_best_model (bool): If True, restores and returns the best model.

    Returns:
        model (nn.Module): The trained model (restored to best state if applicable).
        losses (dict): Dictionary with keys 'train_loss' and 'val_loss' containing loss values per epoch.
    """
    # Instantiate model
    model = SiameseNetwork().to(device)
    model.apply(init_weights)

    # Load datasets
    train_dataset = SiameseDataset(data_dir, pairs_file=pairs_file, augment=augment_train, split=True, mode='train')
    val_dataset = SiameseDataset(data_dir, pairs_file=pairs_file, augment=False, split=True, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf') # Best validation loss
    best_model_state = None  # To store model's best weights
    patience_counter = 0 # Count of consecutive epochs with no improvement
    history = { # Initialize dictionary to track loss per epoch
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    for epoch in range(epochs):
        start_time = time.time()

        #Training
        model.train()
        train_loss = 0.0
        train_acc_total = 0.0
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()

            optimizer.zero_grad() # Reset gradients from previous step
            output = model(img1, img2).squeeze() # Forward pass: get similarity score predictions
            loss = loss_func(output.view(-1), label.view(-1))  # Compute loss between predicted scores and labels
            loss.backward() # Backpropagation
            optimizer.step()  # Update model weights
            train_loss += loss.item() # Accumulate batch loss
            probs = torch.sigmoid(output.view(-1))
            preds = (probs > 0.5).float() # Accuracy
            acc = (preds == label.view(-1)).float().mean().item()
            train_acc_total += acc
        avg_train_loss = train_loss / len(train_loader) # Compute average training loss for the epoch
        avg_train_acc = train_acc_total / len(train_loader)

        #Validation
        model.eval()
        val_loss = 0.0
        val_acc_total = 0.0
        with torch.no_grad(): # Disable gradient computation for validation
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()
                output = model(img1, img2).squeeze()
                loss = loss_func(output.view(-1), label.view(-1))
                val_loss += loss.item()
                probs = torch.sigmoid(output.view(-1))
                preds = (probs > 0.5).float()
                acc = (preds == label.view(-1)).float().mean().item()
                val_acc_total += acc
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc_total / len(val_loader)

        # Store losses for plot
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(avg_train_acc)
        history['val_accuracy'].append(avg_val_acc)

        #Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss # Update best loss
            best_model_state = copy.deepcopy(model.state_dict()) # Save model weights
            patience_counter = 0 # Reset patience counter
        else:
            patience_counter += 1 # Increment if no improvement
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best val loss: {best_val_loss:.4f}")
                if return_best_model and best_model_state:
                    model.load_state_dict(best_model_state)
                break

        #Logging
        if (epoch + 1) % print_every == 0:
            print(f"[{epoch + 1:03d}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f} | "
                  f"Time: {time.time() - start_time:.1f}s")

    # If training ended before best epoch, restore the best weights
    if return_best_model and best_model_state:
        model.load_state_dict(best_model_state)

    return model, history

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None: # safeguard
            nn.init.normal_(m.bias,  mean=0.5, std=0.01)

    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.normal_(m.bias,   mean=0.5, std=0.01)