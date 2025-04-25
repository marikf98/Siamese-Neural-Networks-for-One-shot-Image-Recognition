from train_network import train_siamese_model
import torch
from data_loader import data_loader as SiameseDataset
from evaluate_model import evaluate_siamese_model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model, losses = train_siamese_model(
        data_dir='./lfw2',
        pairs_file='./pairsDevTrain.txt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=16,
        lr=0.0005,
        epochs=20,
        augment_train=True,
        patience=10
    )

    results = evaluate_siamese_model(
        model=model,
        pairs_file='./pairsDevTest.txt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=1,
        threshold=0.5,
        show_examples=True  # see best/worst predictions
    )

