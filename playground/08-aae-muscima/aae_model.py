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


class AaeModel(tf.keras.Model):
    def __init__(self, seed):
        super().__init__()

        self._seed = seed
        self._z_dim = 32
        self._image_shape = (32, 32, 1)

        ### Define the encoder model ###

        encoder_input = tf.keras.layers.Input(
            shape=self._image_shape,
            name="encoder_input"
        )
        # x = downsample(2, encoder_input) # 64x64
        # x = downsample(4, x) # 32x32
        # x = downsample(8, x) # 16x16 (x8) (2048 values)
        x = tf.keras.layers.Flatten()(encoder_input)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        encoder_output = tf.keras.layers.Dense(self._z_dim, activation=None)(x)
        
        self.encoder = tf.keras.Model(
            inputs=encoder_input,
            outputs=encoder_output,
            name="encoder"
        )

        ### Define the decoder model ###
        
        decoder_input = tf.keras.layers.Input(
            shape=(self._z_dim,),
            name="decoder_input"
        )
        x = tf.keras.layers.Dense(512, activation="relu")(decoder_input)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        # x = tf.keras.layers.Dense(
        #     2048 # 16x16 (x8) pixels
        # )(x)
        # x = tf.keras.layers.Reshape(
        #     target_shape=(16, 16, 8)
        # )(x)
        # x = upsample(8, x) # 32x32
        # x = upsample(4, x) # 64x64
        # x = upsample(2, x) # 128x128
        # decoder_output = tf.keras.layers.Conv2D(
        #     1, kernel_size=1,
        #     activation="sigmoid", padding="same"
        # )(x)
        x = tf.keras.layers.Dense(
            np.prod(self._image_shape),
            activation="sigmoid"
        )(x)
        decoder_output = tf.keras.layers.Reshape(
            target_shape=self._image_shape
        )(x)

        self.decoder = tf.keras.Model(
            inputs=decoder_input,
            outputs=decoder_output,
            name="decoder"
        )

        ### Define the discriminator model ###

        discriminator_input = tf.keras.layers.Input(
            shape=(self._z_dim,),
            name="discriminator_input"
        )
        x = tf.keras.layers.Dense(512, activation="relu")(discriminator_input)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        discriminator_output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        self.discriminator = tf.keras.Model(
            inputs=discriminator_input,
            outputs=discriminator_output,
            name="discriminator"
        )

    @tf.function
    def train_step(self, batch):
        images, labels = batch

        with tf.GradientTape() as encoder_tape, \
            tf.GradientTape() as decoder_tape, \
            tf.GradientTape() as discriminator_tape:

            ### Reconstruction phase (update encoder-decoder) ###

            z_encoded = self.encoder(images, training=True)
            decoded_images = self.decoder(z_encoded, training=True)
            reconstruction_loss = self.compiled_loss(
                y_true=images,
                y_pred=decoded_images,
                regularization_losses=self.losses
            )
            
            ### Regularization phase 1 (update discriminator) ###

            # encoder = generator = producer of fake data
            # prior distribution = producer of real data
            z_distribution = tfp.distributions.Normal(
                tf.zeros_like(z_encoded),
                tf.ones_like(z_encoded)
            )
            z_sampled = z_distribution.sample(seed=self._seed)
            
            realness_of_encoded = self.discriminator(z_encoded, training=True)
            realness_of_sampled = self.discriminator(z_sampled, training=True)

            discriminator_loss = self.compiled_loss(
                y_true=tf.zeros_like(realness_of_encoded), # 0 = encoded
                y_pred=realness_of_encoded,
                regularization_losses=self.losses
            ) + self.compiled_loss(
                y_true=tf.ones_like(realness_of_sampled), # 1 = sampled
                y_pred=realness_of_sampled,
                regularization_losses=self.losses
            )

            ### Regularization phase 2 (update generator (encoder)) ###

            # generator loss forces the encoder to generate embeddings
            # that fool the discriminator
            generator_loss = self.compiled_loss(
                y_true=tf.ones_like(realness_of_encoded), # 1 = sampled
                y_pred=realness_of_encoded,
                regularization_losses=self.losses
            )

            ### Aggregate loss functions by models ###

            # Generator loss is reduced, because it was too high and caused the
            # latent space to get compressed into a point or a line during the
            # first epoch. This multiplier keeps it spread out but still
            # within the -1, 1 range.
            generator_multiplier = 0.01

            encoder_loss = reconstruction_loss + generator_loss * generator_multiplier
            decoder_loss = reconstruction_loss
            discriminator_loss = discriminator_loss

        ### gradients & update step & return ###

        encoder_gradients = encoder_tape.gradient(
            encoder_loss, self.encoder.trainable_weights
        )
        decoder_gradients = decoder_tape.gradient(
            decoder_loss, self.decoder.trainable_weights
        )
        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(encoder_gradients, self.encoder.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(decoder_gradients, self.decoder.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_weights)
        )

        return {
            "rec": reconstruction_loss,
            "dis": discriminator_loss,
            "gen": generator_loss
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
        bar = tf.ones([tf.shape(img_head)[0], 4])
        img_tail = tf.concat(tf.unstack(reconstruction, axis=0), axis=0)[:,:,0].numpy()

        final_img = np.concatenate([img_head, bar, img_tail], axis=1)

        # save the image
        tf.keras.utils.save_img(
            "fig/reconstruction-{:02d}.png".format(epoch),
            tf.stack([final_img, final_img, final_img], axis=2) * 255,
            scale=False
        )
