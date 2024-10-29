import torch
from pathlib import Path
import argparse
from model import get_model
from dataloader import get_data_loaders


def evaluate(model, dataloader, device):
    """Evaluate the model on the test dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    return accuracy


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate the model on test data.")
    parser.add_argument('--data_dir', type=str, default=(Path(__file__).resolve().parent.parent / 'data'), help="Path to data directory containing test set")
    parser.add_argument('--weights', type=str, default=(Path(__file__).resolve().parent.parent / 'weights' / 'checkpoint.pth'), help="Path to model weights")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for testing")
    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data only
    print(args.data_dir)
    _, _, test_loader = get_data_loaders(args.data_dir, batch_size=args.batch_size)

    # Load model and weights
    model = get_model().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device)["model_state_dict"])
    print(f"Loaded model weights from {args.weights}")

    # Evaluate the model
    accuracy = evaluate(model, test_loader, device)

    # Optionally, save results
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "test_accuracy.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    print("Evaluation results saved to results/test_accuracy.txt")


if __name__ == "__main__":
    main()
