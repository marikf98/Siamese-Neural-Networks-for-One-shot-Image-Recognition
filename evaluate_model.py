import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_loader import data_loader as SiameseDataset


def evaluate_siamese_model(model, pairs_file, device='cpu', batch_size=16, threshold=0.5, show_examples=True):
    """
    Evaluates the Siamese model and optionally displays best/worst predictions.

    Args:
        model (nn.Module): Trained Siamese model.
        pairs_file (str): Path to the .txt file defining image pairs.
        device (str): Device to evaluate on.
        batch_size (int): Batch size for evaluation.
        threshold (float): Decision threshold.
        show_examples (bool): Whether to visualize best/worst predictions.

    Returns:
        dict: accuracy, scores, labels, predictions.
    """
    test_dataset = SiameseDataset(
        data_dir='./lfw2',
        pairs_file=pairs_file,
        augment=False,
        split=False
    )
    model.eval()
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, label, img1_name, img2_name in loader:
            img1, img2 = img1.to(device), img2.to(device)
            label = label.to(device).float()
            logit = model(img1, img2)
            score = torch.sigmoid(logit).item()
            pred = 1.0 if score > threshold else 0.0

            correct += (pred == label.item())
            total += 1

            results.append({
                'img1': img1.cpu(),
                'img2': img2.cpu(),
                'img1_name': img1_name[0],  # [0] because batch size = 1
                'img2_name': img2_name[0],
                'label': int(label.item()),
                'score': score,
                'pred': int(pred)
            })

    accuracy = correct / total if total > 0 else 0.0

    if show_examples:
        # Sort by score confidence
        correct_preds = [r for r in results if r['label'] == r['pred']]
        incorrect_preds = [r for r in results if r['label'] != r['pred']]

        correct_preds_pos = [r for r in correct_preds if r['label'] == 1 and r['pred'] == 1]
        correct_preds_neg = [r for r in correct_preds if r['label'] == 0 and r['pred'] == 0]
        false_positives = [r for r in incorrect_preds if r['label'] == 0 and r['pred'] == 1]
        false_negatives = [r for r in incorrect_preds if r['label'] == 1 and r['pred'] == 0]

        # Sort by certainty (confidence toward the correct label)
        # Best True Positive: highest score (should be close to 1)
        best_true_positive = max(correct_preds_pos, key=lambda r: r['score'], default=None)
        # Best True Negative: lowest score (should be close to 0)
        best_true_negative = min(correct_preds_neg, key=lambda r: r['score'], default=None)
        # Worst False Negative: lowest score among false negatives (bad because it was too low)
        worst_false_negative = min(false_negatives, key=lambda r: r['score'], default=None)
        # Worst False Positive: highest score among false positives (bad because it was too high)
        worst_false_positive = max(false_positives, key=lambda r: r['score'], default=None)

        print("\nShowing classification examples:")
        show_pair("Best True Positive (Correct Match)", best_true_positive)
        show_pair("Best True Negative (Correct Non-Match)", best_true_negative)
        show_pair("Worst False Negative (Missed Match)", worst_false_negative)
        show_pair("Worst False Positive (Wrong Match)", worst_false_positive)

    return {
        'accuracy': accuracy,
        'results': results
    }


def show_pair(title, pair_data):
    if pair_data is None:
        print(f"{title}: None found.")
        return
    img1 = pair_data['img1'].squeeze().numpy()
    img2 = pair_data['img2'].squeeze().numpy()
    img1_name = pair_data['img1_name']
    img2_name = pair_data['img2_name']
    score = pair_data['score']
    label = pair_data['label']
    pred = pair_data['pred']

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(img1, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title(img1_name, fontsize=8, pad=5)
    axs[1].imshow(img2, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(img2_name, fontsize=8, pad=5)
    fig.suptitle(f"{title}\nScore: {score:.3f} | True: {label} | Pred: {pred}", fontsize=10)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
