# TensorFlow hyperparameter tunning
# https://www.tensorflow.org/tutorials/keras/keras_tuner#instantiate_the_tuner_and_perform_hypertuning

import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt


#
# Create a callback to stop training early after reaching a certain value for the validation loss
#
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

callbacks = [earlyStopping]

#
# Image normalization
#
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def createModel(hp):
    #
    # Create model
    # Tune the number of units in the first Dense layer - Choose an optimal value between 32-512
    #
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=hp_units, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    
    # Compile the model
    # Loss function             : This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
    # Optimizer                 : This is how the model is updated based on the data it sees and its loss function.
    # Metrics                   : Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
    # Tune the learning rate for the optimizer - Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # Display the model's architecture
    model.summary()

    return model

#
# Main flow
#
def main():
    # Load mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize data for better convergence
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Instantiate the tuner and perform hypertuning. This will get optimal hyperparameters for the model.
    tuner = kt.Hyperband(
        createModel,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='hypertunning',
        project_name='introduction')

    tuner.search(
        x_train, 
        y_train, 
        epochs=50, 
        validation_split=0.2, 
        callbacks=callbacks)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print("The hyperparameter search is complete")
    print(f"- Optimal number of units in the first densely-connected layer is {best_hps.get('units')}")
    print(f"- Optimal learning rate for the optimizer is {best_hps.get('learning_rate')}")
   
    # Build the model with the optimal hyperparameters
    hypermodel = tuner.hypermodel.build(best_hps)

    # Train the model on the data for 50 epochs and find best # of epochs
    history = hypermodel.fit(x_train, y_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best number of epochs for this model: %d' % (best_epoch,))

    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)

    # Evaluate the hypermodel on the test data
    eval_result = hypermodel.evaluate(x_test, y_test)
    print("These are the results for best model [test loss, test accuracy]:", eval_result)

#
# Start program
#
if __name__ == '__main__':
    main()