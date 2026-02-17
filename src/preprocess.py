from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split

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

# Transfer learning: put your 2-class dataset here (e.g. transfer_train/dogs, transfer_train/cats)
transfer_train_directory = "transfer_train"
transfer_test_directory = "transfer_test"


def get_transfer_datasets():
    """Load 2-class dataset for transfer learning. Expects transfer_train/ and transfer_test/ with one subdir per class."""
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