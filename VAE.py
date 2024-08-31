def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_vae(input_shape=(28, 28), latent_dim=5):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    x = layers.Reshape((28, 28, 1))(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    outputs = layers.Reshape(input_shape)(outputs)
    decoder = Model(latent_inputs, outputs, name='decoder')

    vae_outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, vae_outputs, name='vae')
    return vae, encoder, decoder

vae, encoder, decoder = create_vae()

def compute_loss(encoder, decoder, x):
    z_mean, z_log_var, z = encoder(x)
    reconstruction = decoder(z)
    mask = tf.math.is_finite(x)
    x_masked = tf.where(mask, x, tf.zeros_like(x))
    reconstruction_masked = tf.where(mask, reconstruction, tf.zeros_like(reconstruction))
    reconstruction_loss = tf.reduce_sum(
        tf.square(x_masked - reconstruction_masked), axis=[1, 2]
    )
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )
    return tf.reduce_mean(reconstruction_loss + kl_loss)
