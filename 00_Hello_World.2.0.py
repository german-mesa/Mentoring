# TensorFlow and tf.keras
# https://www.tensorflow.org/datasets/keras_example

import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Global parameters
NUM_EVALUATIONS = 10
TRAIN_BATCH_SIZE = 128

#
# Early cancellation when reached certain level of accuracy
#
class cancelCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('sparse_categorical_accuracy') > 0.95):
            print("\n\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = [cancelCallback()]

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

    # Check that we have data    
    assert isinstance(ds_train, tf.data.Dataset)

    # Show some examples
    tfds.visualization.show_examples(
        ds_train,
        ds_info
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
    ds_train = ds_train.batch(TRAIN_BATCH_SIZE)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Build evaluation pipeline
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(TRAIN_BATCH_SIZE)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Create and train model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    # Loss function             : This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
    # Optimizer                 : This is how the model is updated based on the data it sees and its loss function.
    # Metrics                   : Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
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
        steps_per_epoch=ds_info.splits['train'].num_examples // (TRAIN_BATCH_SIZE *  NUM_EVALUATIONS),
        epochs=NUM_EVALUATIONS,
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


#
# Start program
#
if __name__ == '__main__':
    main()