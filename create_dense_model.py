## Script create_dense_model.py
import tensorflow as tf
import numpy as np
from tensorflow import keras
DIGIT_CLASSES = 10
RESHAPED = 28 * 28 ## 784 pixels per image
import time
from datetime import datetime


def create_dense_model(hidden_layers=1, hidden_units_per_layer=128):
    """
    create a keras sequential model with the specified
    number of hidden_layers and number of units per layer
    """
    DIGIT_CLASSES = 10
    RESHAPED = 28 * 28 ## 784 pixels per image
    model = tf.keras.models.Sequential()
    count = 0
    
    
    count += 1
    model.add(keras.layers.Dense(units=hidden_units_per_layer,
        input_shape=(RESHAPED,),
        name=f'dense_layer_{count}', activation='relu'))
    
    for i in range(1, hidden_layers):
        count += 1
        model.add(keras.layers.Dense(units=hidden_units_per_layer,
           name=f'dense_layer_{count}', activation='relu'))
        
    count += 1        
    model.add(keras.layers.Dense(DIGIT_CLASSES,
       name=f'dense_layer_{count}', activation='softmax'))

    # summary of the model
    model.summary()
    return model


def run_dense_model(hidden_layers=1, hidden_units_per_layer=128, optimizer='SGD', verbose=1):
    """
    Create and run a dense Keras sequential model over the MNIST training data,
    using the specified numbers of hidden_layers, hidden_units_per_layer, and optimizer.
    Maybe add optional dropout TBD,
    Return values:
    history, training_time, test_accuracy, model
    """
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss',verbose=1,patience=3,min_delta=0.0001)
    
    BATCH_SIZE = 128
    EPOCHS = 400
    VALIDATION_SPLIT = 0.2
    
    ## Load the training and test data
    X_train, X_test, Y_train, Y_test = load_mnist_dataset_flat()
    
    model = create_dense_model(hidden_layers, hidden_units_per_layer, optimizer)
    
    start_time = time.time()

    # compiling the model
    model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    #training the model
    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        verbose=verbose, validation_split=VALIDATION_SPLIT,
                        callbacks=[early_stopping]
                       )
    
    elapsed_time = time.time() - start_time
    
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(f'{hidden_layers=} {hidden_units_per_layer=} {optimizer=}')
    print(f'training elapsed_time = {elapsed_time:10.1f}')
    print('Test accuracy:', test_accuracy)
    
    return history, elapsed_time, test_accuracy, model