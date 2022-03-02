import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


class VaeModel(tf.keras.Model):
    def __init__(self, seed, z_dim, image_shape):
        super().__init__()

        self._seed = seed
        self._z_dim = z_dim
        self._image_shape = image_shape
        self._image_pixels = np.prod(self._image_shape)
        self._z_prior = tfp.distributions.Normal(
            # we 'pull' latent vectors towards this distribution during training
            # we also use this for sampling images during generation
            tf.zeros(self._z_dim),
            tf.ones(self._z_dim)
        )

        ### Define the encoder model ###

        encoder_input = tf.keras.layers.Input(
            shape=self._image_shape,
            name="encoder_input"
        )
        x = tf.keras.layers.Flatten()(encoder_input)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        z_mean_output = tf.keras.layers.Dense(self._z_dim, activation=None)(x)
        z_log_variance_output = tf.keras.layers.Dense(self._z_dim, activation=None)(x)
        
        self.encoder = tf.keras.Model(
            inputs=encoder_input,
            outputs={
                "z_mean": z_mean_output,
                "z_log_variance": z_log_variance_output
            },
            name="encoder"
        )

        ### Define the decoder model ###

        decoder_input = tf.keras.layers.Input(
            shape=(self._z_dim,),
            name="decoder_input"
        )
        x = tf.keras.layers.Dense(500, activation="relu")(decoder_input)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        x = tf.keras.layers.Dense(
            self._image_pixels,
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

    @tf.function
    def train_step(self, batch):
        images, labels = batch

        with tf.GradientTape() as tape:
            
            ### Feed forward ###

            # encode images to get their z-distributions
            encoder_output = self.encoder(images, training=True)
            z_mean = encoder_output["z_mean"]
            z_log_variance = encoder_output["z_log_variance"]

            # sample some actual z-vectors from their z-distributions
            z_distribution = tfp.distributions.Normal(
                z_mean,
                tf.math.exp(z_log_variance / 2)
            )
            z = z_distribution.sample(seed=self._seed)

            # decode z-vectors back to images
            decoded_images = self.decoder(z, training=True)


            ### Compute losses ###

            reconstruction_loss = self.compiled_loss(
                images,
                decoded_images,
                regularization_losses=self.losses
            )

            latent_loss_itemwise = tfp.distributions.kl_divergence(
                z_distribution, # ORDER MATTERS!
                self._z_prior
            )
            latent_loss = tf.math.reduce_mean(latent_loss_itemwise)

            loss = reconstruction_loss * self._image_pixels \
                    + latent_loss * self._z_dim

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return {
            "reconstruction_loss": reconstruction_loss,
            "latent_loss": latent_loss,
            "loss": loss
        }

    def generate(self, epoch, logs):
        GRID = 20

        # random images
        random_images = self.decoder(
            self._z_prior.sample(GRID * GRID, seed=self._seed),
            training=False
        )

        # interpolated images
        if self._z_dim == 2:
            starts = tf.stack([-2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
            ends = tf.stack([2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
        else:
            starts, ends = self._z_prior.sample(GRID, seed=self._seed), self._z_prior.sample(GRID, seed=self._seed)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
        interpolated_images = self.decoder(interpolated_z, training=False)

        # compose the final image
        H, W, C = self._image_shape
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([H, W * GRID, C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)],
            axis=0
        )

        plt.figure(figsize=(10, 20))
        plt.imshow(np.dstack([image, image, image]))
        plt.savefig(
            "fig/manifold-{:02d}.png".format(epoch),
            bbox_inches="tight"
        )
        plt.clf()

    def scatter(self, epoch, images, labels):
        embeddings = self.encoder(
            images,
            training=False
        )["z_mean"]

        fig = plt.figure(figsize=(10, 10))

        for i in range(10):
            plt.scatter(
                x=embeddings[labels == i][:,0],
                y=-embeddings[labels == i][:,1], # flip to match manifold image
                label=str(i)
            )
        
        plt.legend(loc="lower right")
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.savefig(
            "fig/scatter-{:02d}.png".format(epoch),
            bbox_inches="tight"
        )
        plt.clf()
