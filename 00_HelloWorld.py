#
# 00_HelloWorld.py
#
# - Load a prebuilt dataset
# - Build a neural network machine learning model that classifies image numbers
# - Train this neural network
# - Evaluate the accuracy of the model
#
# Full explanation at:
# https://www.tensorflow.org/tutorials/keras/classification#build_the_model
#

# TensorFlow and tf.keras
import tensorflow as tf

print("TensorFlow version:", tf.__version__, "\n")

import matplotlib.pyplot as plt
# Helper libraries
import numpy as np

# Image categories
class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

# Early cancellation when reached certain level of accuracy
class classCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\n\nReached 90% accuracy so cancelling training!")
      # self.model.stop_training = True

def main():
    # Create object mnist 
    mnist = tf.keras.datasets.mnist

    # Load mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data for better convergence
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Plot some figures
    plt.figure(figsize=(10,10))
    for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x_train[i], cmap=plt.cm.binary)
      plt.xlabel(class_names[y_train[i]])
    plt.show()

    # Create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)                           # Ten possible results
    ])

    # Define loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Compile
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    # Train model for 10 epochs
    print("\nTrain model for 10 epochs")
    model.fit(x_train, y_train, epochs=10, callbacks=[classCallback()])

    # Evaluate test dataset
    print("\nEvaluate test dataset")
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc * 100}%')

    # Predictions over test image dataset
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(x_test)
    
    # Print inaccurate predictions
    if input("\nPush <y> in your keyboard to print predictions: ") == 'y':
      for idx, prediction in enumerate(predictions):
        if np.argmax(prediction) != y_test[idx]:
          print(f"Image {idx} was inacurated predicted {class_names[np.argmax(prediction)]} instead of {class_names[y_test[idx]]}")


if __name__ == '__main__':
    main()