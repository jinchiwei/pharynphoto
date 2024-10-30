from pathlib import Path
import torch
from dataloader import get_data_loaders
from model import get_model
from train import train_one_epoch, validate, save_checkpoint
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import csv

def main():
    # config
    dir_mas = Path(__file__).resolve().parent.parent
    dir_data = dir_mas / 'data'
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # results directories
    weights_dir = dir_mas / 'weights'
    results_dir = dir_mas / 'results'
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # dataloaders
    train_loader, val_loader, _ = get_data_loaders(dir_data, batch_size)

    # model, loss function, optimizer
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # lists to store metrics for plotting and saving
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # best val loss tracking
    best_val_loss = float('inf')

    # training loop
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        
        # train and validate for one epoch
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # append metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # save checkpoint if lowest val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = weights_dir / 'checkpoint.pth'
            save_checkpoint(model, optimizer, epoch, filepath=str(checkpoint_path))

    # save training results to a CSV file
    results_csv_path = results_dir / 'training_results.csv'
    with open(results_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy"])
        for epoch in range(num_epochs):
            writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch],
                             train_accuracies[epoch], val_accuracies[epoch]])
    print(f"Training results saved to {results_csv_path}")

    # plot and save loss and accuracy curves
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(results_dir / 'loss_plot.png')
    plt.close()

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(results_dir / 'accuracy_plot.png')
    plt.close()
    print(f"Training and validation curves saved to {results_dir}")


if __name__ == '__main__':
    main()
