import os
import tensorflow as tf
import numpy as np
from .ModelDirectory import ModelDirectory
from .ThresholdingF1Score import ThresholdingF1Score


def _upsample(output_features, x):
    x = tf.keras.layers.UpSampling2D(interpolation="nearest")(x)
    return tf.keras.layers.Conv2D(
        output_features, kernel_size=1,
        activation="relu", padding="same"
    )(x)


def _unet_level(depth, max_depth, inner_features, x):
    level_features = inner_features * (1 << depth)
    
    x = tf.keras.layers.Conv2D(
        level_features, kernel_size=3,
        activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.Conv2D(
        level_features, kernel_size=3,
        activation="relu", padding="same"
    )(x)

    if depth == max_depth: # lowest level stops the recursion
        return x
    
    skip_connection = x
    
    x = tf.keras.layers.MaxPool2D()(x)
    x = _unet_level(depth + 1, max_depth, inner_features, x)
    x = _upsample(output_features=level_features, x=x)
    
    x = tf.concat((skip_connection, x), axis=3)
    
    x = tf.keras.layers.Conv2D(
        level_features, kernel_size=3,
        activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.Conv2D(
        level_features, kernel_size=3,
        activation="relu", padding="same"
    )(x)

    return x


class DenoisingUnetModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.model_directory = ModelDirectory("model")
        self.finished_epochs = 0

        self._segmentation_classes = 1
        self._inner_features = 8

        ### Define the UNet model ###

        # https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png

        # input image
        unet_input = tf.keras.layers.Input(
            shape=(None, None, 1),
            name="unet_input"
        )

        # unet
        x = _unet_level(
            depth=0,
            max_depth=2,
            inner_features=self._inner_features,
            x=unet_input
        )

        # reshape to output classes (sigmoid conv 1x1)
        unet_output = tf.keras.layers.Conv2D(
            self._segmentation_classes, kernel_size=1,
            activation="sigmoid", padding="same"
        )(x)

        self.unet = tf.keras.Model(
            inputs=unet_input,
            outputs=unet_output,
            name="unet"
        )

    def get_config(self):
        return {
            "finished_epochs": self.finished_epochs
        }

    @classmethod
    def from_config(cls, config):
        model = DenoisingUnetModel()
        model.finished_epochs = config["finished_epochs"]
        return model

    @tf.function
    def call(self, inputs, training=None):
        return self.unet(inputs, training=training)

    @tf.function
    def train_step(self, batch):
        # unpack semi-supervised input
        (images, _), (expected_masks, _) = batch

        with tf.GradientTape() as unet_tape:
            
            ### Forward pass ###

            predicted_masks = self.unet(images, training=True)

            ### Compute losses ###
            
            loss = self.compiled_loss(
                y_true=expected_masks,
                y_pred=predicted_masks,
                regularization_losses=self.losses
            )

        ### gradients & update step & return ###

        unet_gradients = unet_tape.gradient(
            loss, self.unet.trainable_weights
        )
        self.optimizer.apply_gradients(
            zip(unet_gradients, self.unet.trainable_weights)
        )

        return {
            "loss": loss
        }


    ########################
    # High-level interface #
    ########################

    @staticmethod
    def load_or_create(model_directory: str):
        dir = ModelDirectory(model_directory)

        model = dir.load_latest_checkpoint(
            custom_objects={
                "DenoisingUnetModel": DenoisingUnetModel,
                "ThresholdingF1Score": ThresholdingF1Score
            }
        )

        if model is None:
            model = DenoisingUnetModel()

        model.model_directory = dir

        model.compile(
            optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.BinaryCrossentropy(from_logits=False),
            metrics=[ThresholdingF1Score()]
        )

        return model

    def perform_training(
        self,
        epochs: int,
        ds_train: tf.data.Dataset,
        ds_validate: tf.data.Dataset
    ):
        self.model_directory.assert_folder_structure()

        VISUALIZE_COUNT = 8
        visualization_batch = ds_validate \
            .unbatch() \
            .batch(VISUALIZE_COUNT) \
            .take(1) \
            .get_single_element()

        if self.finished_epochs == 0:
            self.visualize(0, visualization_batch)

        def _update_finished_epochs(e, l):
            self.finished_epochs = e + 1

        self.fit(
            ds_train,
            epochs=epochs,
            initial_epoch=self.finished_epochs,
            validation_data=ds_validate,
            callbacks=[
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=_update_finished_epochs
                ),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda e, l: self.visualize(
                        e + 1, visualization_batch
                    )
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.model_directory.checkpoint_format_path,
                    monitor="val_loss",
                    verbose=1
                ),
                tf.keras.callbacks.CSVLogger(
                    self.model_directory.metrics_csv_path,
                    separator=',',
                    append=True
                )
            ]
        )

    def visualize(self, epoch, batch):
        images, expected_masks = batch
        predicted_masks = self.call(images, training=False)
        
        def _unstack_channels(batch3d):
            return tf.concat(tf.unstack(batch3d, axis=3), axis=2)

        def _unstack_instances(batch2d):
            return tf.concat(tf.unstack(batch2d, axis=0), axis=0)

        def _unstack_both(batch3d):
            return _unstack_instances(_unstack_channels(batch3d))

        head = _unstack_both(images)
        body = _unstack_both(expected_masks)
        tail = _unstack_both(predicted_masks)
        bar = tf.ones(shape=(head.shape[0], 2), dtype=np.float32)

        img = tf.concat([head, bar, body, bar, tail], axis=1)

        tf.keras.utils.save_img(
            os.path.join(
                self.model_directory.visualizations_path,
                "visualization-{:04d}.png".format(epoch)
            ),
            tf.stack([img, img, img], axis=2) * 255,
            scale=False
        )

    def perform_evaluation(self, ds_test):
        print("Evaluating on the test set:")
        results = self.evaluate(ds_test, return_dict=True)
        print(results)
