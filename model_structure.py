import tensorflow as tf
from tensorflow.keras import Input, Model, layers


def global_min_pooling_2d(x: tf.Tensor) -> tf.Tensor:
    return tf.reduce_min(x, axis=(1, 2), keepdims=True)


def se_block(x: tf.Tensor) -> tf.Tensor:
    out = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(64 * 4, activation="relu")(out)
    out = layers.Dense(64 * 4, activation="sigmoid")(out)
    out = layers.Multiply()([x, layers.Reshape((1, 1, 64 * 4))(out)])
    out = layers.Conv2D(32, kernel_size=1, strides=1, padding="same", activation="relu")(out)
    return layers.Conv2D(1, kernel_size=5, strides=1, padding="same")(out)


def se_max_block(x: tf.Tensor) -> tf.Tensor:
    out = layers.GlobalMaxPooling2D()(x)
    out = layers.Dense(64 * 2, activation="relu")(out)
    out = layers.Dense(64 * 2, activation="sigmoid")(out)
    out = layers.Multiply()([x, layers.Reshape((1, 1, 64 * 2))(out)])
    out = layers.Conv2D(32, kernel_size=1, strides=1, padding="same", activation="relu")(out)
    return layers.Conv2D(1, kernel_size=5, strides=1, padding="same")(out)


def se_min_block(x: tf.Tensor) -> tf.Tensor:
    out = layers.Lambda(global_min_pooling_2d)(x)
    out = layers.Dense(64 * 2, activation="relu")(out)
    out = layers.Dense(64 * 2, activation="sigmoid")(out)
    out = layers.Multiply()([x, layers.Reshape((1, 1, 64 * 2))(out)])
    out = layers.Conv2D(32, kernel_size=1, strides=1, padding="same", activation="relu")(out)
    return layers.Conv2D(1, kernel_size=5, strides=1, padding="same")(out)


def get_generator(
    input_shape1: tuple[int, int, int] = (75, 75, 1),
    input_shape2: tuple[int, int, int] = (75, 75, 1),
    input_shape3: tuple[int, int, int] = (75, 75, 1),
    input_shape4: tuple[int, int, int] = (75, 75, 1),
) -> Model:
    lr_bic = Input(input_shape1, name="lr_bic")
    rf = Input(input_shape2, name="rf")
    dem = Input(input_shape3, name="dem")
    imp = Input(input_shape4, name="imp")
    skip = lr_bic

    lr_bic_feat = layers.Conv2D(64, 9, 1, padding="same", activation="relu")(lr_bic)
    dem_feat = layers.Conv2D(64, 9, 1, padding="same", activation="relu")(dem)
    imp_feat = layers.Conv2D(64, 9, 1, padding="same", activation="relu")(imp)
    rf_feat = layers.Conv2D(64, 9, 1, padding="same", activation="relu")(rf)

    out = layers.Concatenate()([lr_bic_feat, rf_feat, dem_feat, imp_feat])
    rf_dem = layers.Concatenate()([rf_feat, dem_feat])

    rf_dem_max = se_max_block(rf_dem)
    rf_dem_min = se_min_block(rf_dem)
    out = se_block(out)
    out = layers.Add()([out, rf_dem_max, rf_dem_min, skip])

    return Model(inputs=[lr_bic, rf, dem, imp], outputs=out, name="scm_generator")
