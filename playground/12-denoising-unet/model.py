import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Model(tf.keras.Model):
    def __init__(self, seed):
        super().__init__()

        self._seed = seed
        self._segmentation_classes = 1
        self._inner_features = 8

        ### Define the UNet model ###

        # https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png

        def upsample(output_features, x):
            # TODO: try experimenting with different upsamplings
            # a) transposed upconv
            # b) bilinear interpolation
            # ...
            x = tf.keras.layers.UpSampling2D(interpolation="nearest")(x)
            return tf.keras.layers.Conv2D(
                output_features, kernel_size=1,
                activation="relu", padding="same"
            )(x)

        def unet_level(depth, max_depth, x):
            level_features = self._inner_features * (1 << depth)
            
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
            x = unet_level(depth + 1, max_depth, x)
            x = upsample(output_features=level_features, x=x)
            
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

        # input image
        unet_input = tf.keras.layers.Input(
            shape=(None, None, 1),
            name="unet_input"
        )

        # unet
        x = unet_level(depth=0, max_depth=2, x=unet_input)

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

    @tf.function
    def call(self, inputs, training=None):
        return self.unet(inputs, training=training)

    def get_config(self):
        return {"seed": self._seed}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def train_step(self, batch):
        images, expected_masks = batch

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
            "fig/visualization-{:02d}.png".format(epoch),
            tf.stack([img, img, img], axis=2) * 255,
            scale=False
        )
