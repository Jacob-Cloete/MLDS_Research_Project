@register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

@register_keras_serializable()
def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = tf.keras.layers.Conv2D(width, kernel_size=1)(x)
        x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = tf.keras.layers.Conv2D(
            width, kernel_size=3, padding="same", activation=tf.keras.activations.swish
        )(x)
        x = tf.keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.Add()([x, residual])
        return x
    return apply

@register_keras_serializable()
def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply

@register_keras_serializable()
def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply

def get_network(image_size, widths, block_depth):
    noisy_images = tf.keras.Input(shape=(image_size, image_size, 1))
    noise_variances = tf.keras.Input(shape=(1, 1, 1))
    e = tf.keras.layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = tf.keras.layers.UpSampling2D(size=image_size, interpolation="nearest")(e)
    x = tf.keras.layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = tf.keras.layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
        skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest")(x)
        skip = skips.pop()

        # Dynamically adjust padding
        if x.shape[1] < skip.shape[1]:
            x = tf.keras.layers.ZeroPadding2D(((0, skip.shape[1] - x.shape[1]), (0, skip.shape[2] - x.shape[2])))(x)
        elif x.shape[1] > skip.shape[1]:
            diff = x.shape[1] - skip.shape[1]
            x = tf.keras.layers.Cropping2D(((0, diff), (0, diff)))(x)

        x = tf.keras.layers.Concatenate()([x, skip])
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)

    x = tf.keras.layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros")(x)
    return tf.keras.Model([noisy_images, noise_variances], x, name="residual_unet")


class DiffusionModel(tf.keras.Model):
    def __init__(self, network, **kwargs):
        super().__init__(**kwargs)
        self.normalizer = tf.keras.layers.Normalization()
        self.network = network
        self.ema_network = tf.keras.models.clone_model(self.network)

    def get_config(self):
        config = super().get_config()
        config.update({
            "network": tf.keras.layers.serialize(self.network),
            "normalizer": tf.keras.layers.serialize(self.normalizer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        network = tf.keras.layers.deserialize(config.pop("network"))
        model = cls(network)
        model.normalizer = tf.keras.layers.deserialize(config.pop("normalizer"))
        return model

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = tf.keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        diffusion_times = tf.ones((batch_size, 1, 1, 1))  
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        if training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises = network([inputs, noise_rates**2], training=training)
        pred_images = (inputs - noise_rates * pred_noises) / signal_rates
        return pred_images

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, steps):
        batch = initial_noise.shape[0]
        step_size = 1.0 / steps
        next_noisy_images = initial_noise
        for step in range(steps):
            noisy_images = next_noisy_images
            diffusion_times = tf.ones((batch, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    def generate(self, num_images, steps):
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 1))
        generated_images = self.reverse_diffusion(initial_noise, steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    @tf.function
    def train_step(self, images):
        # Create a mask for non-missing values
        mask = tf.not_equal(images, -1)

        # Normalize only the non-missing values
        images_normalized = tf.where(mask, self.normalizer(images), -1)

        noises = tf.random.normal(shape=tf.shape(images))
        batch_size = tf.shape(images)[0]
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = tf.where(mask, signal_rates * images_normalized + noise_rates * noises, -1)

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )
            noise_loss = self.loss(tf.where(mask, noises, 0), tf.where(mask, pred_noises, 0))
            image_loss = self.loss(tf.where(mask, images_normalized, 0), tf.where(mask, pred_images, 0))

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y = data
        mask = tf.not_equal(x, -1)
        x_normalized = tf.where(mask, self.normalizer(x), -1)

        noises = tf.random.normal(shape=tf.shape(x))
        batch_size = tf.shape(x)[0]
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = tf.where(mask, signal_rates * x_normalized + noise_rates * noises, -1)

        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        noise_loss = self.loss(tf.where(mask, noises, 0), tf.where(mask, pred_noises, 0))
        image_loss = self.loss(tf.where(mask, x_normalized, 0), tf.where(mask, pred_images, 0))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        return {m.name: m.result() for m in self.metrics}

def IMPUTE_DIFFUSION(data_missing, diffusion_model, steps):
    n, rows, cols = data_missing.shape
    data_imputed = np.copy(data_missing)

    for i in range(n):
        missing_mask = np.isnan(data_missing[i])
        initial_image = np.where(missing_mask,
                                 0.0,
                                 data_missing[i])
        initial_image = initial_image.reshape(1, rows, cols, 1)
        generated_image = diffusion_model.reverse_diffusion(initial_image, steps)
        generated_image = diffusion_model.denormalize(generated_image)
        data_imputed[i][missing_mask] = generated_image.numpy().reshape(rows, cols)[missing_mask]

    return data_imputed
