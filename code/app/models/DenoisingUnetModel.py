import os
import tensorflow as tf
import numpy as np
from typing import Optional
from .ModelDirectory import ModelDirectory
from .ThresholdingF1Score import ThresholdingF1Score


def _upsample(output_features, x):
    x = tf.keras.layers.UpSampling2D(interpolation="nearest")(x)
    return tf.keras.layers.Conv2D(
        output_features, kernel_size=1,
        activation="relu", padding="same"
    )(x)


def _unet_level(depth, max_depth, inner_features, skip_weight, dropout, x):
    level_features = inner_features * (1 << depth)
    
    x = tf.keras.layers.Conv2D(
        level_features, kernel_size=3,
        activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(
        level_features, kernel_size=3,
        activation="relu", padding="same"
    )(x)

    if depth == max_depth: # lowest level stops the recursion
        return x
    
    skip_connection = x
    
    x = tf.keras.layers.MaxPool2D()(x)
    x = _unet_level(depth + 1, max_depth, inner_features, skip_weight, dropout, x)
    x = _upsample(output_features=level_features, x=x)
    
    # (sz // 2) * 2 is not equal to "sz" if "sz" is not even
    # this fixes that:
    x = tf.image.pad_to_bounding_box(
        x,
        offset_height=0,
        offset_width=0,
        target_height=tf.shape(skip_connection)[1],
        target_width=tf.shape(skip_connection)[2]
    )

    # gate
    skip_connection = skip_connection * skip_weight
    
    # x = tf.concat((skip_connection, x), axis=3)
    x = x + skip_connection # sum instead of concat
    
    x = tf.keras.layers.Conv2D(
        level_features, kernel_size=3,
        activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(
        level_features, kernel_size=3,
        activation="relu", padding="same"
    )(x)

    return x


class DenoisingUnetModel(tf.keras.Model):
    def __init__(
        self,
        input_channels: int = 1,
        segmentation_channels: int = 1,
        denoised_channels: int = 1,
        inner_features: int = 8,
        depth: int = 3,
        unsup_loss_weight: float = 1.0,
        dropout: float = 0.5,
        skip_connection: str = "gated", # one of "none", "gated", "solid"
        fully_supervised: bool = False
    ):
        super().__init__()

        self.model_directory = ModelDirectory("model")
        self.finished_epochs = 0

        self.input_channels = input_channels
        self.segmentation_channels = segmentation_channels
        self.denoised_channels = denoised_channels
        self.inner_features = inner_features
        self.depth = depth
        self.unsup_loss_weight = unsup_loss_weight
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.fully_supervised = fully_supervised

        assert self.skip_connection in ["none", "gated", "solid"]

        ### Define the UNet model ###

        # https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png

        # input image
        unet_input = tf.keras.layers.Input(
            shape=(None, None, self.input_channels),
            name="unet_input"
        )

        skip_weight_input = tf.keras.layers.Input(
            shape=(),
            name="skip_weight_input"
        )
        skip_weight = tf.squeeze(skip_weight_input, axis=0) # hack to get a scalar

        # unet body
        x = _unet_level(
            depth=0,
            max_depth=self.depth,
            inner_features=self.inner_features,
            skip_weight=skip_weight,
            dropout=dropout,
            x=unet_input
        )

        # produce outputs (sigmoid conv 1x1)
        unet_segmentation_output = tf.keras.layers.Conv2D(
            self.segmentation_channels, kernel_size=1,
            activation="sigmoid", padding="same"
        )(x)
        if not self.fully_supervised:
            unet_denoised_output = tf.keras.layers.Conv2D(
                self.denoised_channels, kernel_size=1,
                activation="sigmoid", padding="same"
            )(x)

        # define models
        self.segmentation_unet = tf.keras.Model(
            inputs=[unet_input, skip_weight_input],
            outputs=unet_segmentation_output,
            name="segmentation_unet"
        )
        if not self.fully_supervised:
            self.reconstruction_unet = tf.keras.Model(
                inputs=[unet_input, skip_weight_input],
                outputs=unet_denoised_output,
                name="reconstruction_unet"
            )

    def get_config(self):
        return {
            "finished_epochs": self.finished_epochs,
            "input_channels": self.input_channels,
            "segmentation_channels": self.segmentation_channels,
            "denoised_channels": self.denoised_channels,
            "inner_features": self.inner_features,
            "depth": self.depth,
            "unsup_loss_weight": self.unsup_loss_weight,
            "dropout": self.dropout,
            "skip_connection": self.skip_connection,
            "fully_supervised": self.fully_supervised
        }

    @classmethod
    def from_config(cls, config):
        model = DenoisingUnetModel(
            input_channels=config["input_channels"],
            segmentation_channels=config["segmentation_channels"],
            denoised_channels=config["denoised_channels"],
            inner_features=config["inner_features"],
            depth=config["depth"],
            unsup_loss_weight=config["unsup_loss_weight"],
            dropout=config["dropout"],
            skip_connection=config["skip_connection"],
            fully_supervised=config["fully_supervised"]
        )
        model.finished_epochs = config["finished_epochs"]
        return model

    @tf.function
    def call(self, inputs, training=None):
        if self.skip_connection in ["gated", "solid"]:
            skip_weight = tf.ones(shape=(1,))
        else:
            skip_weight = tf.zeros(shape=(1,))

        return self.segmentation_unet(
            [inputs, skip_weight],
            training=training
        )

    @tf.function
    def call_reconstruct(self, inputs, training=None):
        if self.fully_supervised:
            raise Exception("Cannot do reconstructions on a fully supervised model")
        
        if self.skip_connection in ["solid"]:
            skip_weight = tf.ones(shape=(1,))
        else:
            skip_weight = tf.zeros(shape=(1,))

        return self.reconstruction_unet(
            [inputs, skip_weight],
            training=training
        )

    @tf.function
    def train_step(self, batch):
        # unpack semi-supervised input
        (sup_x, unsup_x), (sup_y_true, unsup_y_true) = batch

        with tf.GradientTape() as tape:
            sup_y_pred = self.call(sup_x, training=True)
            
            if not self.fully_supervised:
                unsup_y_pred = self.call_reconstruct(unsup_x, training=True)
            
            seg_loss = self.compiled_loss(
                y_true=sup_y_true,
                y_pred=sup_y_pred,
                regularization_losses=self.losses
            )

            if not self.fully_supervised:
                noise_loss = self.compiled_loss(
                    y_true=unsup_y_true,
                    y_pred=unsup_y_pred,
                    regularization_losses=self.losses
                )

            if not self.fully_supervised:
                loss = seg_loss + noise_loss * self.unsup_loss_weight # hyperparameter
            else:
                loss = seg_loss

        ### gradients & update step & return ###
        
        gradients = tape.gradient(
            loss, self.trainable_weights
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        ret = {
            "seg_loss": seg_loss
        }
        if not self.fully_supervised:
            ret["noise_loss"] = noise_loss
        return ret


    ########################
    # High-level interface #
    ########################

    @staticmethod
    def load_or_create(model_directory: str, **kwargs):
        dir = ModelDirectory(model_directory)

        model = dir.load_latest_checkpoint(
            custom_objects={
                "DenoisingUnetModel": DenoisingUnetModel,
                "ThresholdingF1Score": ThresholdingF1Score
            }
        )

        if model is None:
            model = DenoisingUnetModel(**kwargs)

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
        ds_validate: tf.data.Dataset,
        save_checkpoints: bool = False,
        early_stop_after: Optional[int] = None
    ):
        self.model_directory.assert_folder_structure()

        visualization_batch = self._build_visualization_batch(ds_train)

        if self.finished_epochs == 0:
            self.visualize(0, visualization_batch)

        def _update_finished_epochs(e, l):
            self.finished_epochs = e + 1

        callbacks = [
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=_update_finished_epochs
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda e, l: self.visualize(
                    e + 1, visualization_batch
                )
            ),
            tf.keras.callbacks.CSVLogger(
                self.model_directory.metrics_csv_path,
                separator=',',
                append=True
            )
        ]

        if save_checkpoints:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.model_directory.checkpoint_format_path,
                    monitor="val_f1_score",
                    verbose=1,
                    save_best_only=True,
                    mode="max"
                ),
            )

        if early_stop_after is not None:
            record = self.model_directory.get_best_epoch_metrics_record()
            baseline = None
            if record is not None:
                if int(record["epoch"]) + 1 == self.finished_epochs:
                    print("Using early stopping baseline from the metrics file.")
                    baseline = record["val_f1_score"]
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_f1_score",
                    verbose=1,
                    patience=early_stop_after,
                    mode="max",
                    baseline=baseline
                )
            )

        self.fit(
            ds_train,
            epochs=epochs,
            initial_epoch=self.finished_epochs,
            validation_data=ds_validate,
            callbacks=callbacks
        )

    def _build_visualization_batch(self, ds_train: tf.data.Dataset):
        VISUALIZE_COUNT = 10

        visu_sup = ds_train.map(lambda x, y: (x[0], y[0])) \
            .unbatch() \
            .batch(VISUALIZE_COUNT) \
            .take(1)

        visu_unsup = ds_train.map(lambda x, y: (x[1], y[1])) \
            .unbatch() \
            .batch(VISUALIZE_COUNT) \
            .take(1)
        
        visu_sup_batch = (None, None)
        for b in visu_sup.as_numpy_iterator():
            visu_sup_batch = b

        visu_unsup_batch = (None, None)
        for b in visu_unsup.as_numpy_iterator():
            visu_unsup_batch = b
        
        return (visu_sup_batch[0], visu_unsup_batch[0]), \
            (visu_sup_batch[1], visu_unsup_batch[1])

    def visualize(self, epoch, batch):
        (sup_x, unsup_x), (sup_y_true, unsup_y_true) = batch
        
        if sup_x is not None:
            sup_y_pred = self.call(sup_x, training=False)
            self.visualize_xy("sup", epoch, sup_x, sup_y_pred, sup_y_true)

        if unsup_x is not None:
            unsup_y_pred = self.call_reconstruct(unsup_x, training=False)
            self.visualize_xy("unsup", epoch, unsup_x, unsup_y_pred, unsup_y_true)

    def visualize_xy(self, name, epoch, x, y_pred, y_true):
        border_color = 1.0
        border_width = 2
        def _add_border(batch3d):
            return tf.image.pad_to_bounding_box(
                image=batch3d - border_color,
                offset_height=0,
                offset_width=0,
                target_height=batch3d.shape[1] + 1,
                target_width=batch3d.shape[2] + 1,
            ) + border_color
        
        def _unstack_channels(batch3d):
            return tf.concat(tf.unstack(batch3d, axis=3), axis=2)

        def _unstack_instances(batch2d):
            return tf.concat(tf.unstack(batch2d, axis=0), axis=0)

        def _unstack_both(batch3d):
            return _unstack_instances(_unstack_channels(_add_border(batch3d)))

        head = _unstack_both(x)
        body = _unstack_both(y_pred)
        tail = _unstack_both(y_true)
        bar = tf.ones(shape=(head.shape[0], border_width), dtype=np.float32)

        img = tf.concat([head, bar, body, bar, tail], axis=1)

        tf.keras.utils.save_img(
            os.path.join(
                self.model_directory.visualizations_path,
                "{}-{:04d}.png".format(name, epoch)
            ),
            tf.stack([img, img, img], axis=2) * 255,
            scale=False
        )

    def perform_evaluation(self, ds_test):
        self.model_directory.assert_folder_structure()
        
        self.visualize_evaluation(ds_test)
        
        print("Evaluating on the test set:")
        results = self.evaluate(ds_test, return_dict=True)
        print(results)

    def visualize_evaluation(self, ds_test):
        for batch, (x, y) in enumerate(ds_test.as_numpy_iterator()):
            print("Visualizing batch", batch)
            y_pred = self.call(x, training=False)
            self.visualize_xy("eval", batch, x, y_pred, y)
