from math import ceil

import torch
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

from model import get_model
from dataloader import get_data_loaders


def evaluate(model, dataloader, device, results_dir):
    """Evaluate the model on the test dataset and save detailed results, including AUC and ROC curve."""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []  # collect probabilities for AUC
    results = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_yes = probabilities[:, 1] * 100  # confidence for "yes pharyngitis" class
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # collect predictions, labels, and probabilities
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probability of "yes" class

            # collect results per sample
            for i in range(labels.size(0)):
                filename = Path(paths[i]).name
                is_correct = "right" if predicted[i] == labels[i] else "wrong"
                results.append(f"{filename} -> {confidence_yes[i]:.2f}% yes -> {is_correct}")

    # calculate overall accuracy
    accuracy = 100 * correct / total
    results.append(f"\nTotal accuracy: {correct}/{total}, {accuracy:.2f}%")

    # calculate AUC
    auc_score = roc_auc_score(all_labels, all_probs)
    results.append(f"AUC: {auc_score:.4f}")

    # save individual sample results and accuracy
    with open(results_dir / "test_accuracy.txt", "w") as f:
        f.write("\n".join(results))
    print("Evaluation results saved to results/test_accuracy.txt")

    # generate and save the confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    display_cm = ConfusionMatrixDisplay(cm, display_labels=['No Pharyngitis', 'Yes Pharyngitis'])
    display_cm.plot(cmap=plt.cm.Blues)
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.close()

    # plot and save the ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(results_dir / "roc_curve.png")
    plt.close()
    print(f"ROC curve saved to {results_dir / 'roc_curve.png'}")

    return accuracy, auc_score


def generate_cam(model, images, labels, device, target_layer='model.layer4'):  # model.layer4.2.conv3
    """Generate Class Activation Maps (CAM) for selected images."""
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)  # Use GradCAM here

    cams = []
    for image in images:
        image = image.to(device)
        
        # perform a forward pass and get the model output
        output = model(image.unsqueeze(0))

        cam = cam_extractor(output.squeeze(0).argmax().item(), output)
        cam_overlay = overlay_mask(to_pil_image(image), to_pil_image(cam[0].squeeze(0), mode='F'), alpha=0.5)
        # cams.append(cam[0].squeeze(0).cpu().numpy())
        cams.append(cam_overlay)

    return cams


def plot_cam(images, cams, class_names, results_dir, idx=0, max_cols=5):
    """Plot and save pre-overlaid CAM images."""
    num_images = len(images)
    num_rows = ceil(num_images / max_cols)  # calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, min(num_images, max_cols), figsize=(15, 5 * num_rows))

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)  # ensure the results directory exists

    # flatten axes for easy iteration if there's more than one row
    axes = axes.flatten() if num_images > 1 else [axes]

    for ax, img, cam_overlay, class_name in zip(axes, images, cams, class_names):
        ax.imshow(cam_overlay)
        ax.axis('off')
        ax.set_title(class_name)

    # turn off any remaining empty axes
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.suptitle("Class Activation Maps")
    plt.savefig(results_dir / f"heatmap_{idx}.png", bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"Class activation heatmaps saved to {results_dir}/heatmap_{idx}.png")


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate the model on test data.")
    parser.add_argument('--data_dir', type=str, default=(Path(__file__).resolve().parent.parent / 'data'), help="Path to data directory containing test set")
    parser.add_argument('--weights', type=str, default=(Path(__file__).resolve().parent.parent / 'weights' / 'checkpoint.pth'), help="Path to model weights")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for testing")
    args = parser.parse_args()

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load test data only
    print(args.data_dir)
    _, _, test_loader = get_data_loaders(args.data_dir, batch_size=args.batch_size)

    # load model and weights
    model = get_model().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device)["model_state_dict"])
    print(f"Loaded model weights from {args.weights}")

    # evaluate the model and generate confusion matrix, AUC, and ROC curve
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    accuracy, auc_score = evaluate(model, test_loader, device, results_dir)
    print(f"Final Test Accuracy: {accuracy:.2f}%, AUC: {auc_score:.4f}")


    # generate CAMs for a few test images
    sample_images, sample_labels, *_ = next(iter(test_loader))
    class_names = ['No Pharyngitis', 'Yes Pharyngitis']
    cams = generate_cam(model, sample_images[:10], sample_labels[:10], device)
    plot_cam(sample_images[:10], cams, [class_names[i] for i in sample_labels[:10]], results_dir, idx=0)


if __name__ == "__main__":
    main()
