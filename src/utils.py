import os
import pathlib
import random
import shutil

from tqdm import tqdm


def split_data(dir_in, dir_out):  
    dir_train = dir_out / 'train'
    dir_val = dir_out / 'val'
    dir_test = dir_out / 'test'

    for folder in [dir_train, dir_val, dir_test]:
        (folder / 'no_pharyngitis').mkdir(parents=True, exist_ok=True)
        (folder / 'yes_pharyngitis').mkdir(parents=True, exist_ok=True)

    ratio_train = 0.7
    ratio_val = 0.15
    ratio_test = 0.15

    def split_and_copy_images(class_folder):
        images = list(class_folder.glob('*'))
        random.shuffle(images)

        # calculate the number of images per split
        train_cutoff = int(len(images) * ratio_train)
        val_cutoff = int(len(images) * (ratio_train + ratio_val))

        # split images into train, val, and test sets
        train_images = images[:train_cutoff]
        val_images = images[train_cutoff:val_cutoff]
        test_images = images[val_cutoff:]

        # copy images to respective directories
        for img in tqdm(train_images, desc=f"Copying {class_folder.name} images to train"):
            shutil.copy(img, dir_train / class_folder.name / img.name)
        for img in tqdm(val_images, desc=f"Copying {class_folder.name} images to val"):
            shutil.copy(img, dir_val / class_folder.name / img.name)
        for img in tqdm(test_images, desc=f"Copying {class_folder.name} images to test"):
            shutil.copy(img, dir_test / class_folder.name / img.name)

    # split both 'no' and 'yes' class folders
    for class_name in tqdm(['no_pharyngitis', 'yes_pharyngitis'], desc='Classes:'):
        class_folder = dir_in / class_name
        split_and_copy_images(class_folder)

    print('Data has been split into 70-15-15 train-val-test ratio.')


def main():
    dir_in = pathlib.Path(__file__).resolve().parent.parent
    dir_in = pathlib.Path(os.getcwd()) / 'data' / 'raw'
    dir_out = pathlib.Path(os.getcwd()) / 'data'
    split_data(dir_in, dir_out)


if __name__ == '__main__':
    main()
