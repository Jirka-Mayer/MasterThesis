import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def downsample(features, x):
    x = tf.keras.layers.Conv2D(
        features, kernel_size=3,
        activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    return x


def upsample(features, x):
    x = tf.keras.layers.UpSampling2D(interpolation="nearest")(x)
    x = tf.keras.layers.Conv2D(
        features, kernel_size=3,
        activation="relu", padding="same"
    )(x)
    return x


class Model(tf.keras.Model):
    def __init__(self, seed):
        super().__init__()

        self._seed = seed
        self._image_shape = (32, 32, 1)
        self._z_dim = (16, 16, 4)

        ### Define the encoder model ###

        encoder_input = tf.keras.layers.Input(
            shape=self._image_shape,
            name="encoder_input"
        )
        x = encoder_input
        
        x = tf.keras.layers.Conv2D( # 32x32x1 -> 16x16x32
            filters=32, kernel_size=2, strides=[2, 2],
            activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.Conv2D( # 16x16x32 -> 16x16x4
            filters=4, kernel_size=1,
            activation="relu", padding="same"
        )(x)
        
        encoder_output = x
        self.encoder = tf.keras.Model(
            inputs=encoder_input,
            outputs=encoder_output,
            name="encoder"
        )

        ### Define the decoder model ###
        
        decoder_input = tf.keras.layers.Input(
            shape=self._z_dim,
            name="decoder_input"
        )
        x = decoder_input
        
        x = tf.keras.layers.Conv2D( # 16x16x4 -> 16x16x32
            filters=32, kernel_size=1,
            activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.UpSampling2D()(x) # 16x16x32 -> 32x32x32
        x = tf.keras.layers.Conv2D( # 32x32x32 -> 32x32x1
            filters=1, kernel_size=4,
            activation="sigmoid", padding="same"
        )(x)

        decoder_output = x
        self.decoder = tf.keras.Model(
            inputs=decoder_input,
            outputs=decoder_output,
            name="decoder"
        )

    @tf.function
    def train_step(self, batch):
        images, labels = batch

        with tf.GradientTape() as encoder_tape, \
            tf.GradientTape() as decoder_tape:

            ### Reconstruction phase (update encoder-decoder) ###

            z_encoded = self.encoder(images, training=True)
            decoded_images = self.decoder(z_encoded, training=True)
            reconstruction_loss = self.compiled_loss(
                y_true=tf.concat([images, labels], axis=3),
                y_pred=decoded_images,
                regularization_losses=self.losses
            )

            ### Sparsity penalty ###

            # sparsity penalty forces the encoder to create sparse embeddings
            # sparsity_loss = tf.reduce_mean(z_encoded)

            ### Aggregate loss functions by models ###

            # Tunable hyperparameter
            # sparsity_multiplier = 0.1

            encoder_loss = reconstruction_loss #+ sparsity_loss * sparsity_multiplier
            decoder_loss = reconstruction_loss

        ### gradients & update step & return ###

        encoder_gradients = encoder_tape.gradient(
            encoder_loss, self.encoder.trainable_weights
        )
        decoder_gradients = decoder_tape.gradient(
            decoder_loss, self.decoder.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(encoder_gradients, self.encoder.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(decoder_gradients, self.decoder.trainable_weights)
        )

        return {
            "rec": reconstruction_loss
            # "spr": sparsity_loss
        }

    def visualize_reconstruction(self, epoch, images, labels):
        z = self.encoder(
            images,
            training=False
        )
        reconstruction = self.decoder(
            z,
            training=False
        )
        
        img_head = tf.concat(tf.unstack(images, axis=0), axis=0)[:,:,0].numpy()
        #img_head2 = tf.concat(tf.unstack(labels, axis=0), axis=0)[:,:,0].numpy()
        
        bar = tf.ones([tf.shape(img_head)[0], 4])

        z = tf.image.resize(z, [32, 32], method="nearest")
        z = tf.concat(tf.unstack(z, axis=3), axis=2)
        img_body = tf.concat(tf.unstack(z, axis=0), axis=0).numpy()

        reconstruction = tf.concat(tf.unstack(reconstruction, axis=3), axis=2)
        img_tail = tf.concat(tf.unstack(reconstruction, axis=0), axis=0).numpy()

        final_img = np.concatenate([img_head, bar, img_body, bar, img_tail], axis=1)

        # save the image
        tf.keras.utils.save_img(
            "fig/reconstruction-{:02d}.png".format(epoch),
            tf.stack([final_img, final_img, final_img], axis=2) * 255,
            scale=False
        )
