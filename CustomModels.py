import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter


class GAN(keras.Model):
    """A GAN Model, modified from Keras official WGAN example.

    Args:
        keras (_type_): _description_
    """
    def __init__(
        self,
        discriminator,
        generator,
        discriminator_extra_steps=3,
        gp_weight=10.0
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, min_max_loss, g_losses, g_losses_weights, g_metrics):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.min_max_loss = min_max_loss
        self.g_losses = g_losses
        self.g_losses_weights = g_losses_weights
        self.g_metrics = g_metrics

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):

        data = data_adapter.expand_1d(data)
        inputs, real_images, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        
        fake_images = self.generator(inputs, training=True)
        
        for i in range(self.d_steps):
        
            with tf.GradientTape() as tape:
                
                # Get the logits for the fake images
                discriminator_output_4fake = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                discriminator_output_4real = self.discriminator(real_images, training=True)
                # Loss for the discriminator maximize (Them the sign is negative)
                min_max_loss = -self.min_max_loss(discriminator_output_4real, discriminator_output_4fake)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = min_max_loss + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            fake_images = self.generator(inputs, training=True)
            # Get the discriminator logits for fake images
            discriminator_output_4fake = self.discriminator(fake_images, training=True)
            
            # Calculate the loss for the generator minimize
            
            min_max_loss = tf.reduce_mean(self.min_max_loss(discriminator_output_4real, discriminator_output_4fake))
            min_max_loss = tf.squeeze(min_max_loss)
            
            # Calculate the other generator losses
        
            other_losses_dict = {loss.__class__.__name__ : tf.reduce_mean(loss(real_images, fake_images)) for loss in self.g_losses}
            other_losses = list(other_losses_dict.values())
            
            weighted_losses = [loss*weight for loss, weight in zip(other_losses, self.g_losses_weights)]
            weighted_losses_mean = tf.squeeze(tf.reduce_mean(weighted_losses, axis = 0))
            
            total_loss = min_max_loss + weighted_losses_mean
            
        gen_gradient = tape.gradient(total_loss, self.generator.trainable_variables)
        
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        #generating losses and metrics dict
        results = {self.min_max_loss.__name__: min_max_loss, "gp":gp}
        results.update(other_losses_dict)
        results.update({metric.__class__.__name__ : tf.reduce_mean(metric(real_images, fake_images)) for metric in self.g_metrics})
        
        return results
    
    def test_step(self, data):
        """The logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.

        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.

        Arguments:
        data: A nested structure of `Tensor`s.

        Returns:
        A `dict` containing values that will be passed to
        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
        values of the `Model`'s metrics are returned.
        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        batch_size = tf.shape(y)[0]

        fake_images = self.generator(x)
        
        discriminator_output_4fake = self.discriminator(fake_images)
        discriminator_output_4real = self.discriminator(y)
      
        gp = self.gradient_penalty(batch_size, y, fake_images)
        min_max_loss =  tf.reduce_mean(self.min_max_loss(discriminator_output_4real, discriminator_output_4fake))
        other_losses_dict = {loss.__class__.__name__ : tf.reduce_mean(loss(y, fake_images)) for loss in self.g_losses}
        metrics = {metric.__class__.__name__ : tf.reduce_mean(metric(y, fake_images)) for metric in self.g_metrics}
        
        results = {self.min_max_loss.__name__: min_max_loss, "gp":gp}
        results.update(other_losses_dict)
        results.update(metrics)
        
        return results
        
    def save(self, folder_path):
        self.generator.save(folder_path + f"/{self.generator.name}.tf")
        self.discriminator.save(folder_path + f"/{self.discriminator.name}.tf")
        
    @property
    def name(self):
        """
            The name of the gan is the name of the generator and discriminator
        """
        return f"{self.generator.name}_{self.discriminator.name}"
        
    def count_params(self):
        """
           Returns a dict containing the generator and discriminator parameters count
        """
        return {"gan": self.generator.count_params(), "discriminator": self.discriminator.count_params()}
    
    
    def call(self, inputs, training=True, mask=None):
        """
            Call for the generator
        """
        return self.generator(inputs)