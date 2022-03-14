import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


class AaeModel(tf.keras.Model):
    def __init__(self, seed, supervised_ratio):
        super().__init__()

        self._seed = seed
        self._z_dim = 2
        self._image_shape = (28, 28, 1)
        self._label_count = 10
        self._image_pixels = np.prod(self._image_shape)
        self._z_prior = tfp.distributions.Normal(
            # used only for sampling during "generate" call,
            # not during training due to the batch dimension missing
            loc=tf.zeros(self._z_dim),
            scale=tf.ones(self._z_dim)
        )
        self._supervised_ratio = supervised_ratio

        ### Define the encoder model ###

        encoder_input = tf.keras.layers.Input(
            shape=self._image_shape,
            name="encoder_input"
        )
        x = tf.keras.layers.Flatten()(encoder_input)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
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
        x = tf.keras.layers.Dense(500, activation="relu")(decoder_input)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        x = tf.keras.layers.Dense(
            self._image_pixels * (self._label_count + 1),
            activation="sigmoid"
        )(x)
        decoder_output = tf.keras.layers.Reshape(
            target_shape=(28, 28, self._label_count + 1)
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
        x = tf.keras.layers.Dense(500, activation="relu")(discriminator_input)
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
            expected_masks = self.images_to_expected_masks(images, labels)
            sample_weight = self.batch_sample_weight_tensor(tf.shape(images)[0])
            reconstruction_loss = self.compiled_loss(
                y_true=expected_masks * sample_weight, # to make sure the masked data IS zero
                y_pred=decoded_images * sample_weight, # to make it equal to gold and kill gradients
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
            "disc": discriminator_loss,
            "gen": generator_loss
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
    #     plt.clf()

    @tf.function
    def images_to_expected_masks(self, images, labels):
        """Converts images with known labels to the desired segmentation masks"""
        @tf.function
        def image_to_mask(args):
            image, label = args
            channel_stack = tf.concat(
                [image] + [
                    image * tf.cond(
                        tf.constant(i, dtype=tf.int64) == label,
                        lambda: 1.0,
                        lambda: 0.0
                    )
                    for i in range(self._label_count)
                ],
                axis=2
            )
            return channel_stack, label

        masks, _ = tf.map_fn(
            fn=image_to_mask,
            elems=(images, labels)
        )
        return masks

    @tf.function
    def batch_sample_weight_tensor(self, batch_size):
        labelled_count = tf.cast(
            tf.cast(batch_size, tf.float32) * self._supervised_ratio,
            tf.int32
        )
        keep = tf.ones(shape=(labelled_count, 28, 28, self._label_count))
        mask = tf.zeros(shape=(batch_size - labelled_count, 28, 28, self._label_count))
        segmentation_channels = tf.concat([keep, mask], axis=0)
        reconstruction_channel = tf.ones(shape=(batch_size, 28, 28, 1))
        return tf.concat([reconstruction_channel, segmentation_channels], axis=3)

    def visualize_sample(self, epoch, images, labels):
        z = self.encoder(
            images,
            training=False
        )
        masks = self.decoder(
            z,
            training=False
        )

        expected_masks = self.images_to_expected_masks(images, labels)

        def unroll_batch_and_channel_dims(images): # 4D tensor -> 2D tensor
            # unroll batch vertically, keep channels
            x = tf.concat(tf.unstack(images, axis=0), axis=0)
            # unroll channels horizontally
            return tf.concat(tf.unstack(x, axis=2), axis=1)

        img_inputs = unroll_batch_and_channel_dims(images).numpy()
        img_inputs[:,27] = 1.0
        img_masks = unroll_batch_and_channel_dims(masks).numpy()
        img_expectations = unroll_batch_and_channel_dims(expected_masks).numpy()
        img_expectations[:,0] = 1.0
        final_img = np.concatenate([img_inputs, img_masks, img_expectations], axis=1)
        
        # save the image
        plt.figure(figsize=(10, 10))
        plt.imshow(final_img)
        plt.savefig(
            "fig/masks-{:02d}.png".format(epoch),
            bbox_inches="tight"
        )
        plt.clf()

    def scatter(self, epoch, images, labels):
        embeddings = self.encoder(
            images,
            training=False
        )

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
