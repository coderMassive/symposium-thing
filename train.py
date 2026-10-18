import keras

train = keras.utils.image_dataset_from_directory(
    directory="chest_xray/train/",
    labels="inferred",
    label_mode="binary",
    batch_size=8,
    image_size=(1024, 1024),
    shuffle=True
)

test = keras.utils.image_dataset_from_directory(
    directory="chest_xray/test/",
    labels="inferred",
    label_mode="binary",
    batch_size=8,
    image_size=(1024, 1024),
    shuffle=True
)

model = keras.Sequential([
    keras.Input(shape=(1024, 1024, 3)),
    keras.layers.Rescaling(1.0 / 255),
    keras.layers.Conv2D(32, kernel_size=13, strides=2, padding="same", activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.Conv2D(12, kernel_size=1, activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(48, kernel_size=11, padding="same", activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.Conv2D(16, kernel_size=1, activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(48, kernel_size=7, padding="same", activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.Conv2D(20, kernel_size=1, activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(96, kernel_size=5, activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.Conv2D(384, kernel_size=1, activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)), 
    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation=keras.layers.LeakyReLU(negative_slope=0.1), kernel_regularizer=keras.regularizers.L2(0.01)),
    keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.L2(0.01)),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

checkpoint_filepath = 'best_model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

model.fit(
    train,
    validation_data=test,
    epochs=50,
    callbacks=[model_checkpoint_callback]
)
