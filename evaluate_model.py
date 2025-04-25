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
        for img1, img2, label in loader:
            img1, img2 = img1.to(device), img2.to(device)
            label = label.to(device).float()
            score = model(img1, img2).item()
            pred = 1.0 if score > threshold else 0.0

            correct += (pred == label.item())
            total += 1

            results.append({
                'img1': img1.cpu(),
                'img2': img2.cpu(),
                'label': int(label.item()),
                'score': score,
                'pred': int(pred)
            })

    accuracy = correct / total if total > 0 else 0.0

    if show_examples:
        # Sort by score confidence
        correct_preds = [r for r in results if r['label'] == r['pred']]
        incorrect_preds = [r for r in results if r['label'] != r['pred']]

        # Sort by certainty (confidence toward the correct label)
        best_correct = max(correct_preds, key=lambda r: r['score'] if r['label'] == 1 else 1 - r['score'], default=None)
        worst_correct = min(correct_preds, key=lambda r: r['score'] if r['label'] == 1 else 1 - r['score'], default=None)
        best_fp = max(incorrect_preds, key=lambda r: r['score'], default=None)  # false positive = score too high
        worst_fn = min(incorrect_preds, key=lambda r: r['score'], default=None)  # false negative = score too low


        print("\nShowing classification examples:")
        show_pair("Best Correct Match", best_correct)
        show_pair("Worst Correct Match", worst_correct)
        show_pair("Best False Positive", best_fp)
        show_pair("Worst False Negative", worst_fn)

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
    score = pair_data['score']
    label = pair_data['label']
    pred = pair_data['pred']

    fig, axs = plt.subplots(1, 2, figsize=(4, 2))
    axs[0].imshow(img1, cmap='gray')
    axs[1].imshow(img2, cmap='gray')
    axs[0].axis('off')
    axs[1].axis('off')
    fig.suptitle(f"{title}\nScore: {score:.3f} | True: {label} | Pred: {pred}")
    plt.tight_layout()
    plt.show()
