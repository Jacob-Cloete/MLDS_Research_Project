# data
diffusion_steps = 20
image_size = 28

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# optimization
batch_size = 64
num_epochs = 5
learning_rate = 1e-3
weight_decay = 1e-4
ema = 0.999
embedding_max_frequency = 1000.0
embedding_dims = 64

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
    conditional_images = tf.keras.Input(shape=(image_size, image_size, 1))
    confidence = tf.keras.Input(shape=(image_size, image_size, 1))

    e = tf.keras.layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = tf.keras.layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = tf.keras.layers.Concatenate()([noisy_images, conditional_images, confidence])
    x = tf.keras.layers.Conv2D(widths[0], kernel_size=1)(x)
    x = tf.keras.layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = ResidualBlock(width)(x)
        skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        x = tf.keras.layers.Concatenate()([x, skips.pop()])
        x = ResidualBlock(width)(x)

    x = tf.keras.layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros")(x)
    return tf.keras.Model([noisy_images, noise_variances, conditional_images, confidence], x, name="residual_unet")

@register_keras_serializable()
class DiffusionModel(tf.keras.Model):
    def __init__(self, network):
        super().__init__()
        self.normalizer = tf.keras.layers.Normalization()
        self.network = network
        self.ema_network = tf.keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = tf.keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, conditional_images, confidence, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network([noisy_images, noise_rates**2, conditional_images, confidence], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, conditional_images, confidence, steps):
        batch = tf.shape(initial_noise)[0]
        next_noisy_images = initial_noise
        for step in range(steps):
            noisy_images = next_noisy_images
            diffusion_times = tf.ones((batch, 1, 1, 1)) - step / steps
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, conditional_images, confidence, training=False
            )
            next_diffusion_times = diffusion_times - 1 / steps
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    def denormalize(self, images):
        return self.normalizer.mean + images * tf.sqrt(self.normalizer.variance)

    def generate(self, conditional_images, mask, steps):
        batch_size = tf.shape(conditional_images)[0]
        initial_noise = tf.random.normal(shape=(batch_size, image_size, image_size, 1))
        confidence = tf.cast(mask, tf.float32)
        conditional_images_normalized = self.normalizer(conditional_images)
        generated_images = self.reverse_diffusion(initial_noise, conditional_images_normalized, confidence, steps)
        return self.denormalize(generated_images)

    def create_mask(self, shape, missing_rate=0.8):
        return tf.cast(tf.random.uniform(shape) > missing_rate, tf.float32)

    def calculate_confidence(self, mask, diffusion_times, total_steps):
        return mask + (1 - diffusion_times / total_steps) * (1 - mask)

    def create_conditional_images(self, images, noisy_images, mask):
        return mask * images + (1 - mask) * noisy_images

    def train_step(self, images):
        images_normalized = self.normalizer(images)

        noises = tf.random.normal(shape=tf.shape(images))
        batch_size = tf.shape(images)[0]
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images_normalized + noise_rates * noises

        # Create mask and conditional images
        mask = self.create_mask(tf.shape(images))
        conditional_images = self.create_conditional_images(images_normalized, noisy_images, mask)
        confidence = self.calculate_confidence(mask, diffusion_times, 1.0)

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, conditional_images, confidence, training=True
            )
            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(images_normalized, pred_images)

            # Add penalty for near-zero imputation
            zero_penalty = 0.1 * tf.reduce_mean(tf.square(1.0 - tf.abs(pred_images)))

            total_loss = noise_loss + image_loss + zero_penalty

        gradients = tape.gradient(total_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # Update EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        images_normalized = self.normalizer(images)

        noises = tf.random.normal(shape=tf.shape(images))
        batch_size = tf.shape(images)[0]
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images_normalized + noise_rates * noises

        # Create mask and conditional images for validation
        mask = self.create_mask(tf.shape(images))
        conditional_images = self.create_conditional_images(images_normalized, noisy_images, mask)
        confidence = self.calculate_confidence(mask, diffusion_times, 1.0)

        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, conditional_images, confidence, training=False
        )
        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images_normalized, pred_images)

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        return {m.name: m.result() for m in self.metrics}
