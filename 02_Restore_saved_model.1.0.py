#
# 02_Restore_saved_model.py
#
# - Load a prebuilt model
# - Evaluate the accuracy of the model
#

# TensorFlow and tf.keras
import tensorflow as tf
print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import os
import numpy as np


#
# Main flow
#
def main():
    # Load mnist dataset
    # http://yann.lecun.com/exdb/mnist/
    #
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize data for better convergence
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create model either directly or loading weights from other run
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "models", "mnist.h5"))
 
    # Evaluate test dataset
    print("\nEvaluate test dataset")
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print("\nTest accuracy: {:5.2f}%".format(100 * test_acc))

    # Predictions over test image dataset
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(x_test)


#
# Start program
#
if __name__ == '__main__':
    main()
