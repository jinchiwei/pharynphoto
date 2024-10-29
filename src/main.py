from pathlib import Path
import torch
from dataloader import get_data_loaders
from model import get_model
from train import train_one_epoch, validate, save_checkpoint
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

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

    # lists to store losses for plotting
    train_losses = []
    val_losses = []

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

        # append losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save checkpoint if lowest val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # checkpoint_path = weights_dir / f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_path = weights_dir / f'checkpoint.pth'
            save_checkpoint(model, optimizer, epoch, filepath=str(checkpoint_path))

    # plot train/val loss graph
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # save plot
    plot_path = results_dir / 'loss_plot.png'
    plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':
    main()
