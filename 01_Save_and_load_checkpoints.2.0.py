# TensorFlow and tf.keras
# https://www.tensorflow.org/datasets/keras_example
#
import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import os
import matplotlib.pyplot as plt


#
# Create a callback that saves the model's weights
#
checkCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(os.getcwd(), "training", "cp.ckpt"),
    save_weights_only=True,
    verbose=1
)

callbacks = [checkCallback]

#
# Image normalization
#
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


#
# Main flow
#
def main():
    # Load dataset
    # https://www.tensorflow.org/datasets/catalog/mnist
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Build training pipeline
    # - tf.data.Dataset.map     : TFDS provide images of type tf.uint8, while the model expects tf.float32. Therefore, you need to normalize images.
    # - tf.data.Dataset.cache   : As you fit the dataset in memory, cache it before shuffling for a better performance.
    # - tf.data.Dataset.shuffle : For true randomness, set the shuffle buffer to the full dataset size.
    # - tf.data.Dataset.batch   : Batch elements of the dataset after shuffling to get unique batches at each epoch.
    # - tf.data.Dataset.prefetch: It is good practice to end the pipeline by prefetching for performance.
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Build evaluation pipeline
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Create and train model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    # Loss function             : This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
    # Optimizer                 : This is how the model is updated based on the data it sees and its loss function.
    # Metrics                   : Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Display the model's architecture
    model.summary()

    # Upload existing weights for model
    if input("\nPush <y> to load previous weights: ") == 'y':
        latest = tf.train.latest_checkpoint(os.path.join(os.getcwd(), "training"))
        model.load_weights(latest)

    # Fit the model
    history =  model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
        callbacks=callbacks
    )
    
    # Plot loss & accuracy
    # https://matplotlib.org/
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.suptitle('Model Loss & Accuracy')

    # Plot each graph, and manually set the y tick values
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_ylabel('loss')

    axs[1].plot(history.history['sparse_categorical_accuracy'])
    axs[1].plot(history.history['val_sparse_categorical_accuracy'])
    axs[1].set_ylabel('accuracy')

    plt.show()

    # Save the entire model to a HDF5 file.
    if input("\nPush <y> to save the entire model: ") == 'y':
        model.save(os.path.join(os.getcwd(), "models", "mnist.h5"))

#
# Start program
#
if __name__ == '__main__':
    main()