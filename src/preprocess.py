import os
from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split

_src_dir = os.path.dirname(os.path.abspath(__file__))

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='grayscale', #CHANGED FROM RGB
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='grayscale', #CHANGED FROM RGB
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

transfer_train_directory = os.path.join(_src_dir, "transfer_train")
transfer_test_directory = os.path.join(_src_dir, "transfer_test")


def get_transfer_datasets():
    if not os.path.isdir(transfer_train_directory):
        raise FileNotFoundError(
            f"Transfer train folder not found: {transfer_train_directory}\n"
            "Create src/transfer_train/ with two subfolders (e.g. dogs/, cats/) and put images in each."
        )
    if not os.path.isdir(transfer_test_directory):
        raise FileNotFoundError(
            f"Transfer test folder not found: {transfer_test_directory}\n"
            "Create src/transfer_test/ with the same two subfolders and put test images in each."
        )
    train_subdirs = [d for d in os.listdir(transfer_train_directory) if os.path.isdir(os.path.join(transfer_train_directory, d)) and not d.startswith(".")]
    if len(train_subdirs) != 2:
        raise ValueError(
            f"transfer_train must have exactly 2 class subfolders, found {len(train_subdirs)}: {train_subdirs}. "
            "E.g. transfer_train/dogs/ and transfer_train/cats/ (each with images inside)."
        )
    print("transfer train dataset:")
    train_dataset, validation_dataset = image_dataset_from_directory(
        transfer_train_directory,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47,
    )
    print("transfer test dataset:")
    test_dataset = image_dataset_from_directory(
        transfer_test_directory,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    return train_dataset, validation_dataset, test_dataset