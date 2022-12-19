# TensorFlow and tf.keras
# https://www.tensorflow.org/datasets/keras_example
# https://matplotlib.org/
#
import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


#
# Early cancellation when reached certain level of accuracy
#
class classCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('sparse_categorical_accuracy') > 0.99):
            print("\n\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

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
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Build training pipeline
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Display the model's architecture
    model.summary()

    # Fit the model
    history =  model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
        callbacks=[classCallback()]
    )
    
    # Plot loss & accuracy
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


#
# Start program
#
if __name__ == '__main__':
    main()