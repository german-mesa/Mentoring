#
# 00_HelloWorld.py
#
# - Load a prebuilt dataset
# - Build a neural network machine learning model that classifies image numbers
# - Train this neural network
# - Evaluate the accuracy of the model
#

# TensorFlow and tf.keras
import tensorflow as tf
print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Image categories
class_names = ['Zero', 'One', 'Two', 'Three',
               'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

#
# Early cancellation when reached certain level of accuracy
#
class classCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95):
            print("\n\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

#
# Create & compile our model
#
def create_model():
    # Create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),    # 784 = 28 x 28   0
        # 128             784 x 128 + 128
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),                     # 128             0
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
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize data for better convergence
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Plot some figures
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show()

    # Create model
    model = create_model()

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
                print(
                    f"Image {idx} was inacurated predicted {class_names[np.argmax(prediction)]} instead of {class_names[y_test[idx]]}")

#
# Start program
#
if __name__ == '__main__':
    main()
