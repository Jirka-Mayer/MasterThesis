from numpy import identity
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


class UnetModel(tf.keras.Model):
    def __init__(self, seed):
        super().__init__()

        self._seed = seed
        self._segmentation_classes = 10
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
            shape=(28, 28, 1),
            name="unet_input"
        )

        # unet
        x = unet_level(depth=0, max_depth=2, x=unet_input)

        # reshape to output classes (sigmoid conv 1x1)
        unet_output = tf.keras.layers.Conv2D(
            self._segmentation_classes, kernel_size=1,
            activation="sigmoid", padding="same"
        )(x)

        # define the model (in two output variants)
        self.unet = tf.keras.Model(
            inputs=unet_input,
            outputs=unet_output,
            name="unet"
        )

        # print the model
        tf.keras.utils.plot_model(self.unet, "model.png", show_shapes=True)

        # TODO:
        # image -> seg mask
        # seg mask -> seg mask, identity on the same model
        #   (but only with the k-most prominent masks)
        #   a) on the mask as is
        #   b) on the mask with noise where zero

    @tf.function
    def train_step(self, batch):
        images, labels = batch

        @tf.function
        def build_mask_inputs_slice(batch_slice):
            class_prominences = tf.math.reduce_sum(batch_slice, axis=(0, 1))
            largest_class_index = tf.math.argmax(class_prominences)
            return batch_slice[:, :, largest_class_index]

        @tf.function
        def build_mask_outputs_slice(batch_slice):
            class_prominences = tf.math.reduce_sum(batch_slice, axis=(0, 1))
            largest_class_index = tf.math.argmax(class_prominences)

            batch_slice_transposed = tf.transpose(batch_slice, [2, 0, 1])
            outputs_slice_transposed = tf.tensor_scatter_nd_update(
                tf.zeros(
                    shape=batch_slice_transposed.shape,
                    dtype=batch_slice_transposed.dtype
                ),
                indices=[[largest_class_index]],
                updates=[batch_slice_transposed[largest_class_index]]
            )
            outputs_slice = tf.transpose(outputs_slice_transposed, [1, 2, 0])

            return outputs_slice

        with tf.GradientTape() as tape:
            
            ### Segmentation pass ###

            pass_1_input = images
            pass_1_output = self.unet(images, training=True)

            ### Mask training pass ###

            # pass_2_input = tf.map_fn(
            #     build_mask_inputs_slice,
            #     pass_1_output
            # )
            # pass_2_expected_output = tf.map_fn(
            #     build_mask_outputs_slice,
            #     pass_1_output
            # )
            # pass_2_actual_output = self.unet(pass_2_input, training=True)

            ### Compute losses ###

            # all masks should reconstruct the input image
            identity_loss = self.compiled_loss(
                y_true=tf.concat(
                    [pass_1_input * 0] + [pass_1_input for _ in range(self._segmentation_classes - 1)],
                    axis=3
                ),
                y_pred=pass_1_output,
                sample_weight=None,
                regularization_losses=self.losses
            )
            # identity_loss = tf.losses.BinaryCrossentropy(from_logits=False)(
            #     y_true=tf.stack(
            #         [pass_1_input for _ in range(self._segmentation_classes)],
            #         axis=3
            #     ),
            #     y_pred=pass_1_output
            # )

            # masks should (when combined) reconstruct the image
            # reconstruction_loss = tf.losses.BinaryCrossentropy(from_logits=True)(
            #     y_true=pass_1_input,
            #     y_pred=tf.expand_dims(
            #         tf.reduce_sum(pass_1_output_logits, axis=3),
            #         axis=3
            #     )
            # )

            # mask should be classified into its own class
            # mask_stabilization_loss = tf.losses.BinaryCrossentropy(from_logits=False)(
            #     y_true=pass_2_expected_output,
            #     y_pred=pass_2_actual_output
            # )

            # the proper mask should light up the most
            # supervised_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)(
            #     y_true=labels,
            #     y_pred=tf.reduce_mean(pass_1_output_logits, axis=(1, 2))
            # )
            
            # def prepare_sup_output(args):
            #     input_slice, label = args
            #     channel_stack = tf.stack(
            #         [
            #             input_slice
            #             # input_slice * tf.cond(
            #             #     tf.constant(i, dtype=tf.int64) == label,
            #             #     lambda: 0.0,
            #             #     lambda: 1.0
            #             # )
            #             for i in range(self._segmentation_classes)
            #         ],
            #         axis=2
            #     )
            #     return channel_stack, label

            # supervised_output, _ = tf.map_fn(
            #     fn=prepare_sup_output,
            #     elems=(pass_1_input, labels)
            # )
            
            # supervised_loss = tf.losses.BinaryCrossentropy(from_logits=False)(
            #     y_true=supervised_output,
            #     y_pred=pass_1_output
            # )

            #loss = reconstruction_loss + mask_stabilization_loss
            loss = identity_loss

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return {
            # "reconstruction_loss": reconstruction_loss,
            # "mask_stabilization_loss": mask_stabilization_loss,
            # "identity_loss": identity_loss,
            # "supervised_loss": supervised_loss,
            "loss": loss
        }

    def scatter(self, epoch, images, labels):
        masks = self.unet(images, training=False)
        
        graphics = []
        for i in range(images.shape[0]):
            image = images[i]
            image_masks = masks[i]
            image_masks_transposed = tf.transpose(image_masks, [2, 0, 1])
            graphic = tf.concat(
                [image[:,:,0]] + [t for t in image_masks_transposed],
                axis=1
            ).numpy()
            graphic[:,27] = 1 # draw a separation line
            graphics.append(graphic)

        final_graphic = np.concatenate(graphics, axis=0)

        # save the image
        plt.figure(figsize=(10, 10))
        plt.imshow(np.dstack([final_graphic, final_graphic, final_graphic]))
        plt.savefig(
            "fig/masks-{:02d}.png".format(epoch),
            bbox_inches="tight"
        )
        plt.clf()
