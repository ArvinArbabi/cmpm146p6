import numpy as np
from preprocess import get_datasets
from models.basic_model import BasicModel
from models.model import Model
from config import image_size
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time

# CHANGED: grayscale input (1 channel)
input_shape = (image_size[0], image_size[1], 1)
categories_count = 3

models = {
    'basic_model': BasicModel,
}

def plot_history(history):
    h = history if isinstance(history, dict) else history.history
    acc = h['accuracy']
    val_acc = h['val_accuracy']
    loss = h['loss']
    val_loss = h['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("results/name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/name_of_your_model.npy', allow_pickle=True).item()
    # plot_history(history)
    #
    # Your code should change the number of epochs
    # I changed it from 1 to 15 to 30
    epochs = 100

    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()

    name = 'basic_model'
    model_class = models[name]

    print('* Training {} for up to {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    model.print_summary()

    model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
    best_model_path = 'results/{}.keras'.format(model_name)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=15,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_model_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    history = model.train_model(train_dataset, validation_dataset, epochs,
                                callbacks=callbacks)

    print('* Evaluating {}'.format(name))
    model.evaluate(test_dataset)

    print('* Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))

    model.save_model(best_model_path)
    np.save('results/{}.npy'.format(model_name), history.history)

    print('* Best model saved as {}'.format(best_model_path))
    plot_history(history)
