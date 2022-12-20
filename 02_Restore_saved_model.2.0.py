# TensorFlow and tf.keras
# https://www.tensorflow.org/datasets/keras_example
#
import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import os
import numpy as np

# Image categories
class_names = [
    'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

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
    assert isinstance(ds_test, tf.data.Dataset)

    # Build evaluation pipeline
    # - tf.data.Dataset.map     : TFDS provide images of type tf.uint8, while the model expects tf.float32. Therefore, you need to normalize images.
    # - tf.data.Dataset.cache   : As you fit the dataset in memory, cache it before shuffling for a better performance.
    # - tf.data.Dataset.batch   : Batch elements of the dataset after shuffling to get unique batches at each epoch.
    # - tf.data.Dataset.prefetch: It is good practice to end the pipeline by prefetching for performance.
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Create model either directly or loading weights from other run
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "models", "mnist.h5"))
 
    # Predictions over test image dataset
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = model.predict(ds_test, batch_size=128)

    for prediction in predictions:
        print(f"Predicted value is {tf.math.argmax(prediction)}")


#   
# Start program
#
if __name__ == '__main__':
    main()
