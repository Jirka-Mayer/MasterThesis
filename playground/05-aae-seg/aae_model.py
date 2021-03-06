import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


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


def unet_level(depth, max_depth, inner_features, x):
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
    x = unet_level(depth + 1, max_depth, inner_features, x)
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


class AaeModel(tf.keras.Model):
    def __init__(self, seed):
        super().__init__()

        self._seed = seed
        self._z_dim = 4 # number of segmentation channels
        # NOTE: due to softmax, the first channel is the mask negative
        self._inner_features = 8 # unet base channel depth
        self._image_shape = (28, 28, 1)
        self._image_pixels = np.prod(self._image_shape)
        self._z_prior = tfp.distributions.Normal(
            # used only for sampling during "generate" call,
            # not during training due to the batch dimension missing
            loc=tf.zeros(self._z_dim),
            scale=tf.ones(self._z_dim)
        )
        self._unet_encoder = True

        ### Define the encoder model ###

        encoder_input = tf.keras.layers.Input(
            shape=self._image_shape,
            name="encoder_input"
        )

        if self._unet_encoder:
            x = unet_level(
                depth=0,
                max_depth=2,
                inner_features=self._inner_features,
                x=encoder_input
            )
            encoder_output = tf.keras.layers.Conv2D(
                self._z_dim, kernel_size=1,
                activation="softmax", padding="same"
            )(x)
        else:
            x = tf.keras.layers.Flatten()(encoder_input)
            x = tf.keras.layers.Dense(500, activation="relu")(x)
            x = tf.keras.layers.Dense(500, activation="relu")(x)
            x = tf.keras.layers.Dense(
                self._image_pixels * self._z_dim,
                activation="sigmoid"
            )(x)
            encoder_output = tf.keras.layers.Reshape(
                target_shape=(28, 28, self._z_dim)
            )(x)
        
        self.encoder = tf.keras.Model(
            inputs=encoder_input,
            outputs=encoder_output,
            name="encoder"
        )

        ### Define the decoder model ###
        
        decoder_input = tf.keras.layers.Input(
            shape=(28, 28, self._z_dim),
            name="decoder_input"
        )
        x = tf.keras.layers.Flatten()(decoder_input)
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

        discriminator_input = tf.keras.layers.Input(
            shape=(28, 28, self._z_dim),
            name="discriminator_input"
        )
        x = tf.keras.layers.Flatten()(discriminator_input)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        discriminator_output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        self.discriminator = tf.keras.Model(
            inputs=discriminator_input,
            outputs=discriminator_output,
            name="discriminator"
        )

    @tf.function
    def sample_z(self, images):
        """Samples the Z prior distribution from a given batch of images"""
        # TODO: SAMPLE SHAPES ONLY (coz softmax)!!! AND COMPLETELY RANDOMIZE!!!
        # TODO: also make the first image a negative
        return tf.concat(
            [1.0 - tf.random.shuffle(images, seed=self._seed)] + \
            [tf.random.shuffle(images, seed=self._seed + 1 + i) for i in range(self._z_dim - 1)],
            axis=3
        )
        
        # LEGACY CODE:
        # it needs these images to sample the geometry
        # and to determine the batch size
        # batch_size = tf.shape(images)[0]
        # z_distribution = tfp.distributions.Uniform(
        #     low=tf.fill([batch_size, self._z_dim], 0.0),
        #     high=tf.fill([batch_size, self._z_dim], 1.0)
        # )
        # z_sampled_color = z_distribution.sample(seed=self._seed)
        # z_sampled_geometry = tf.random.shuffle(images, seed=self._seed)
        # z_sampled = tf.map_fn(
        #     fn=lambda x: tf.concat(
        #         [x[0] * x[1][i] for i in range(self._z_dim)],
        #         axis=2
        #     ),
        #     elems=[z_sampled_geometry, z_sampled_color],
        #     fn_output_signature=tf.TensorSpec([28, 28, self._z_dim])
        # )
        # return z_sampled

    def visualize_z(self, z):
        """Visualizes a Z value as a grayscale image"""
        x = tf.concat(tf.unstack(z, axis=3), axis=2)
        x = tf.concat(tf.unstack(x, axis=0), axis=0)
        # plt.imshow(x)
        # plt.show()
        return x

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
            z_sampled = self.sample_z(images)

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

            ### Segmentation regularization - the last n-1 masks add up to input ###

            segmentation_loss = self.compiled_loss(
                y_true=images,
                y_pred=tf.expand_dims(
                    # sum all masks except for the first one, otherwise we would
                    # always get ones (due to the softmax normalization)
                    tf.reduce_sum(z_encoded[:,:,:,1:], axis=3),
                    axis=3
                ),
                regularization_losses=self.losses
            )

            nonempty_loss = -tf.reduce_mean(z_encoded[:,:,:,1:])

            ### Aggregate loss functions by models ###

            # Generator loss is reduced, because it was too high and caused the
            # latent space to get compressed into a point or a line during the
            # first epoch. This multiplier keeps it spread out but still
            # within the -1, 1 range.
            generator_multiplier = 0.01
            segmentation_multiplier = 1.0
            nonempty_multiplier = 1.0

            encoder_loss = reconstruction_loss + \
                generator_loss * generator_multiplier + \
                segmentation_loss * segmentation_multiplier + \
                nonempty_loss * nonempty_multiplier
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
            "nem": nonempty_loss,
            "segm": segmentation_loss,
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

    def visualize_sample(self, epoch, images, labels):
        z = self.encoder(
            images,
            training=False
        )
        reconstruction = self.decoder(
            z,
            training=False
        )
        img = self.visualize_z(z).numpy()
        img_head = tf.concat(tf.unstack(images, axis=0), axis=0)[:,:,0].numpy()
        img_head[:,27] = 1.0
        img_tail = tf.concat(tf.unstack(reconstruction, axis=0), axis=0)[:,:,0].numpy()
        img_tail[:,0] = 1.0

        final_img = np.concatenate([img_head, img, img_tail], axis=1)
        
        # save the image
        plt.figure(figsize=(10, 10))
        plt.imshow(final_img)
        plt.savefig(
            "fig/embedding-{:02d}.png".format(epoch),
            bbox_inches="tight"
        )
        plt.clf()

    def scatter(self, epoch, images, labels):
        # TODO: convert image to color and plot
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
