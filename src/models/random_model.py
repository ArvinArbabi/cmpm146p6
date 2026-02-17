import numpy as np
from models.model import Model
from tensorflow.keras import Sequential, layers, Model as KerasModel
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        base = load_model("results/best_model.keras")
        base_cut = KerasModel(inputs=base.input, outputs=base.layers[-2].output)
        for layer in base_cut.layers:
            layer.trainable = False
        self._randomize_layers(base_cut)
        self.model = Sequential([
            base_cut,
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(categories_count, activation="softmax"),
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    @staticmethod
    def _randomize_layers(model):
        for layer in model.layers:
            w = layer.get_weights()
            if w:
                layer.set_weights([np.random.randn(*arr.shape).astype(np.float32) * 0.01 for arr in w])
