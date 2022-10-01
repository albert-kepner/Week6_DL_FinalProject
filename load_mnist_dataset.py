import tensorflow as tf
from tensorflow import keras
def load_mnist_dataset(train_size=10000, flatten=True):
    """
    Load the MNIST data set consisting of handwritten single digit images 0-9
    formatted as 28 x 28 pixels with one channel (black and white)
    60000 images are for training
    10000 images are for testing
    This function reshapes the images as flattened 784 byte arrays, if needed
    The image bytes are normalized as float in [0,1]
    Y_train and Y_test are one-hot encoded for the 10 digit classes
    returns X_train, X_test, Y_train, Y_test
    """
    DIGIT_CLASSES = 10
    mnist = keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    #
    if flatten:
        #X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
        RESHAPED = 784
        X_train = X_train.reshape(60000, RESHAPED)
        X_test = X_test.reshape(10000, RESHAPED)
    else:
        # Retain 2D image shape for CNN
        X_train = X_train.reshape((60000, 28, 28, 1))
        X_test = X_test.reshape((10000, 28, 28, 1))

            
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #normalize in [0,1]
    X_train /= 255
    X_test /= 255

    #one-hot
    Y_train = tf.keras.utils.to_categorical(Y_train, DIGIT_CLASSES)
    Y_test = tf.keras.utils.to_categorical(Y_test, DIGIT_CLASSES)
    ## Split off a smaller training sample of specified size and use
    ## the remainder of the training data as a validation set
    X_valid = X_train[train_size:]
    Y_valid = Y_train[train_size:]
    X_train = X_train[:train_size]
    Y_train = Y_train[:train_size]
    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'validation samples')
    print(X_test.shape[0], 'test samples')
    
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test