## script create_baseline_model.py
import tensorflow as tf
import numpy as np
from tensorflow import keras
import time
from datetime import datetime

def create_baseline_model():
    DIGIT_CLASSES = 10
    RESHAPED = 28 * 28 ## 784 pixesl per image
    #build the model
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Dense(DIGIT_CLASSES,
                                 input_shape=(RESHAPED,),
                                 name='dense_layer',
                                 activation='softmax'))
    return model

def run_baseline_model(optimizer='SGD', verbose=0):
    """
    Create and run a dense Keras sequential model over the MNIST training data,
    using the specified numbers of hidden_layers, hidden_units_per_layer, and optimizer.
    Maybe add optional dropout TBD,
    Return values:
    history, training_time, test_accuracy, model
    """
    from keras.callbacks import EarlyStopping
    from load_mnist_dataset_flat import load_mnist_dataset_flat
    
    early_stopping = EarlyStopping(monitor='val_loss',verbose=1,patience=3,min_delta=0.0001)
    
    BATCH_SIZE = 128
    EPOCHS = 400
    VALIDATION_SPLIT = 0.2
    
    ## Load the training and test data
    X_train, X_test, Y_train, Y_test = load_mnist_dataset_flat()
    
    model = create_baseline_model()
    
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
    print(f'training elapsed_time = {elapsed_time:10.1f}')
    print('Test accuracy:', test_accuracy)
    
    return  elapsed_time, test_accuracy