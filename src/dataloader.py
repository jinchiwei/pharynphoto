from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths."""
    def __getitem__(self, index):
        # original tuple of (image, label)
        original_tuple = super().__getitem__(index)
        # get image file path
        path = self.imgs[index][0]
        # make a new tuple that includes the original and the path
        return original_tuple + (path,)


def get_data_loaders(data_dir, batch_size=32):
    # define transformations with augmentations for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
        transforms.RandomRotation(15, expand=False),           # Randomly rotate the image by 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Random crop and resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # define transformations for validation and test (no augmentations)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load datasets from pre-split folders
    train_dataset = datasets.ImageFolder(root=Path(data_dir) / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=Path(data_dir) / "val", transform=eval_transform)
    test_dataset = ImageFolderWithPaths(root=Path(data_dir) / 'test', transform=eval_transform)

    # create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
