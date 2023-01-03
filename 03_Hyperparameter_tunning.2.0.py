# TensorFlow hyperparameter tunning - Work in progress
# https://www.tensorflow.org/tutorials/keras/keras_tuner#instantiate_the_tuner_and_perform_hypertuning

import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__, "\n")

# Helper libraries
import keras.backend as kb
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


#
# Make iterator
#
def make_iterator(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()

    with kb.get_session().as_default() as sess:
        while True:
            *inputs, labels = sess.run(next_val)
            yield inputs, labels


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

    # Build evaluation pipeline
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()

    # Create TensorFlow Iterator object and wrap it in a generator
    itr_train = make_iterator(ds_train)
    itr_valid = make_iterator(ds_test)

    # Instantiate the tuner and perform hypertuning. This will get optimal hyperparameters for the model.
    tuner = kt.Hyperband(
        createModel,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='hypertunning',
        project_name='introduction')

    tuner.search(
        itr_train,
        validation_data=itr_valid,
        epochs=50,  
        callbacks=callbacks)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print("The hyperparameter search is complete")
    print(f"- Optimal number of units in the first densely-connected layer is {best_hps.get('units')}")
    print(f"- Optimal learning rate for the optimizer is {best_hps.get('learning_rate')}")
   
    # Build the model with the optimal hyperparameters
    hypermodel = tuner.hypermodel.build(best_hps)

    # Train the model on the data for 50 epochs and find best # of epochs
    history = hypermodel.fit(itr_train, epochs=50, validation_data=itr_valid)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best number of epochs for this model: %d' % (best_epoch,))

    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(itr_train, epochs=best_epoch, validation_data=itr_valid)

    # Evaluate the hypermodel on the test data
    eval_result = hypermodel.evaluate(itr_valid)
    print("These are the results for best model [test loss, test accuracy]:", eval_result)

#
# Start program
#
if __name__ == '__main__':
    main()