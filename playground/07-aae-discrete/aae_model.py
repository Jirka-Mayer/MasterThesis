import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


class AaeModel(tf.keras.Model):
    def __init__(self, seed):
        super().__init__()

        self._seed = seed
        self._z_dim = 2
        self._z_discrete_dim = 10
        self._image_shape = (28, 28, 1)
        self._image_pixels = np.prod(self._image_shape)
        self._z_prior = tfp.distributions.Normal(
            # used only for sampling during "generate" call,
            # not during training due to the batch dimension missing
            loc=tf.zeros(self._z_dim),
            scale=tf.ones(self._z_dim)
        )

        ### Define the encoder model ###

        encoder_input = tf.keras.layers.Input(
            shape=self._image_shape,
            name="encoder_input"
        )
        x = tf.keras.layers.Flatten()(encoder_input)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        encoder_output = {
            "continuous": tf.keras.layers.Dense(self._z_dim, activation=None)(x),
            "discrete": tf.keras.layers.Dense(self._z_discrete_dim, activation="softmax")(x)
        }
        
        self.encoder = tf.keras.Model(
            inputs=encoder_input,
            outputs=encoder_output,
            name="encoder"
        )

        ### Define the decoder model ###
        
        decoder_input = {
            "continuous": tf.keras.layers.Input(
                shape=(self._z_dim,),
                name="decoder_input"
            ),
            "discrete": tf.keras.layers.Input(
                shape=(self._z_discrete_dim,),
                name="decoder_input_discrete"
            )
        }
        x = tf.concat(
            [decoder_input["continuous"], decoder_input["discrete"]],
            axis=1
        )
        x = tf.keras.layers.Dense(500, activation="relu")(x)
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

        ### Define the discriminator model ###

        discriminator_input = {
            "continuous": tf.keras.layers.Input(
                shape=(self._z_dim,),
                name="descriminator_input"
            ),
            "discrete": tf.keras.layers.Input(
                shape=(self._z_discrete_dim,),
                name="descriminator_input_discrete"
            )
        }
        x = tf.concat(
            [discriminator_input["continuous"], discriminator_input["discrete"]],
            axis=1
        )
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
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
            z_distribution_continuous = tfp.distributions.Normal(
                tf.zeros_like(z_encoded["continuous"]),
                tf.ones_like(z_encoded["continuous"])
            )
            z_distribution_discrete = tfp.distributions.OneHotCategorical(
                logits=tf.ones_like(z_encoded["discrete"])
            )
            z_sampled_continuous = z_distribution_continuous.sample(seed=self._seed)
            z_sampled_discrete = z_distribution_discrete.sample(seed=self._seed)
            z_sampled = {
                "continuous": z_sampled_continuous,
                "discrete": z_sampled_discrete
            }
            
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
            "reconstruction_loss": reconstruction_loss,
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss
        }

    # def generate(self, epoch, logs):
    #     GRID = 20

    #     # random images
    #     random_images = self.decoder(
    #         self._z_prior.sample(GRID * GRID, seed=self._seed),
    #         training=False
    #     )

    #     # interpolated images
    #     if self._z_dim == 2:
    #         starts = tf.stack([-2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
    #         ends = tf.stack([2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
    #     else:
    #         starts, ends = self._z_prior.sample(GRID, seed=self._seed), self._z_prior.sample(GRID, seed=self._seed)
    #     interpolated_z = tf.concat(
    #         [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
    #     interpolated_images = self.decoder(interpolated_z, training=False)

    #     # compose the final image
    #     H, W, C = self._image_shape
    #     image = tf.concat(
    #         [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
    #         [tf.zeros([H, W * GRID, C])] +
    #         [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)],
    #         axis=0
    #     )

    #     plt.figure(figsize=(10, 20))
    #     plt.imshow(np.dstack([image, image, image]))
    #     plt.savefig(
    #         "fig/manifold-{:02d}.png".format(epoch),
    #         bbox_inches="tight"
    #     )
    #     plt.close()

    def generate(self, epoch, logs):
        row_images = []
        for zi in range(self._z_discrete_dim):
            z_discrete = [0] * self._z_discrete_dim
            z_discrete[zi] = 1
            cell_images = []
            for si in range(8):
                z_style = np.random.standard_normal(size=2)
                if si == 0:
                    z_style = [0.0, 0.0]
                img = self.decoder(
                    {
                        "continuous": np.reshape(z_style, (1, self._z_dim)),
                        "discrete": np.reshape(z_discrete, (1, self._z_discrete_dim))
                    },
                    training=False
                )[0, :, :, 0]
                cell_images.append(img)
            row_images.append(np.concatenate(cell_images, axis=1))
        final_img = np.concatenate(row_images, axis=0)

        plt.figure(figsize=(10, 20))
        plt.imshow(np.dstack([final_img, final_img, final_img]))
        plt.savefig(
            "fig/sampling-{:02d}.png".format(epoch),
            bbox_inches="tight"
        )
        plt.close()

    def scatter(self, epoch, images, labels):
        embeddings = self.encoder(
            images,
            training=False
        )

        ### 2D continuous dim scatter ###

        plt.figure(figsize=(10, 10))

        for i in range(10):
            plt.scatter(
                x=embeddings["continuous"][labels == i][:,0],
                y=-embeddings["continuous"][labels == i][:,1], # flip to match manifold image
                label=str(i)
            )
        
        plt.legend(loc="lower right")
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.savefig(
            "fig/scatter-{:02d}.png".format(epoch),
            bbox_inches="tight"
        )
        plt.close()

        ### 10D discrete dim scatter ###

        fig, ax = plt.subplots()

        embedding_bins = np.zeros(shape=(self._z_discrete_dim, 10))

        for bi in range(images.shape[0]):
            for zi in range(self._z_discrete_dim):
                # discrete counting
                # if embeddings["discrete"][bi, zi] > 0.5:
                #     embedding_bins[zi, labels[bi]] += 1
                # smooth counting
                embedding_bins[zi, labels[bi]] += embeddings["discrete"][bi, zi]

        bottom = [0] * self._z_discrete_dim
        for li in range(10):
            ax.bar(
                ["Z" + str(zi) for zi in range(self._z_discrete_dim)],
                embedding_bins[:, li],
                bottom=bottom,
                label=str(li)
            )
            for zi in range(self._z_discrete_dim):
                bottom[zi] += embedding_bins[zi, li]

        ax.legend(loc="lower right")
        plt.savefig(
            "fig/scatter-discrete-{:02d}.png".format(epoch),
            bbox_inches="tight"
        )
        plt.close()
