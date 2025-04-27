from train_network import train_siamese_model
import torch
from evaluate_model import evaluate_siamese_model, plot_training_history
# uncoment for local
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model, history = train_siamese_model(
        data_dir='./lfw2',
        pairs_file='./pairsDevTrain.txt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=16,
        lr=0.0005,
        epochs=2,
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

    plot_training_history(history)

