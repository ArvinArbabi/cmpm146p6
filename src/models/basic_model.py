from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers.legacy import Adam


class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255.0),

            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),

            layers.Conv2D(32, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
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
