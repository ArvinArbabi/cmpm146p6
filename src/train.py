import numpy as np
from preprocess import get_datasets
from models.basic_model import BasicModel
from models.model import Model
from config import image_size
import matplotlib.pyplot as plt
import time

# CHANGED: grayscale input (1 channel)
input_shape = (image_size[0], image_size[1], 1)
categories_count = 3

models = {
    'basic_model': BasicModel,
}

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
    # model = Model.load_model("name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/name_of_your_model.npy', allow_pickle='TRUE').item()
    # plot_history(history)
    #
    # Your code should change the number of epochs
    # I changed it from 1 to 15 to 30
    epochs = 30

    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()

    # Optional one-time sanity check (uncomment if you want)
    # for x, y in train_dataset.take(1):
    #     print("batch x shape:", x.shape)  # should be (batch, H, W, 1)
    #     print("batch y shape:", y.shape)  # should be (batch, 3)

    name = 'basic_model'
    model_class = models[name]

    print('* Training {} for {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    model.print_summary()

    history = model.train_model(train_dataset, validation_dataset, epochs)

    print('* Evaluating {}'.format(name))
    model.evaluate(test_dataset)

    print('* Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))

    model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
    filename = 'results/{}.keras'.format(model_name)

    model.save_model(filename)
    np.save('results/{}.npy'.format(model_name), history)

    print('* Model saved as {}'.format(filename))
    plot_history(history)
