#
# 01_Save_and_load_checkpoints.py
#
# - Load a prebuilt dataset
# - Build a neural network machine learning model that classifies image numbers
# - Train this neural network
# - Save checkpoints and load weights on execution
# - Evaluate the accuracy of the model
#

# TensorFlow and tf.keras
import tensorflow as tf
print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Image categories
class_names = ['Zero', 'One', 'Two', 'Three',
               'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

#
# Create a callback that saves the model's weights
#
checkCallback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.getcwd(), "training", "cp.ckpt"),
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks = [checkCallback]

#
# Create & compile our model
#
def create_model():
    # Create model
    model = tf.keras.models.Sequential([
        # 784 = 28 x 28   0
        tf.keras.layers.Flatten(input_shape=(28, 28)),    
        # 128             784 x 128 + 128
        tf.keras.layers.Dense(128, activation='relu'),
         # 128             0
        tf.keras.layers.Dropout(0.2),                    
        # 10              128 x 10 + 10
        tf.keras.layers.Dense(10)
    ])

    # Define loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Display the model's architecture
    model.summary()

    return model

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
    model = create_model()
    if input("\nPush <y> to load previous weights: ") == 'y':
        print("\nUpload existing weights for model")
        
        latest = tf.train.latest_checkpoint(os.path.join(os.getcwd(), "training"))
        model.load_weights(latest)

    # Train model for 10 epochs - saves checkpoints during training
    print("\nTrain model for 10 epochs")
    model.fit(x_train, y_train, epochs=10, callbacks=callbacks)

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
