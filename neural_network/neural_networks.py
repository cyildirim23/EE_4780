import tensorflow as tf
import numpy as np
import os

from neural_network.image_processing import cv_image_to_gray, cv_load_image

def network_create(outputs: int = 10,
        input_shape: tuple = (128, 128),
        hidden_layers: int = 2, 
        neurons: int = 128,
        conv_size: int = None,
        conv_depth: int = None,
        activation_function_hidden = tf.nn.relu,
        activation_function_output = tf.nn.softmax,
        optimizer_algo = 'adam',
        loss_algo = 'sparse_categorical_crossentropy',
        metrics_list = ['accuracy']
    ) -> tf.keras.models.Sequential:
    #Create Sequential Neural Network
    model = tf.keras.models.Sequential()

    #Input Layer to Neural Network.
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    if conv_size is None:

        #Flatten Layer for Dense Layer
        model.add(tf.keras.layers.Flatten())

        #Dense Layer with neurons. Use relu activation function
        for _ in range(hidden_layers):
            model.add(tf.keras.layers.Dense(neurons, activation=activation_function_hidden))
    else:
        #Convolution2D layer.
        for _ in range(conv_depth):
            model.add(tf.keras.layers.Conv1D(filters=conv_size, kernel_size=3, padding='same', activation='relu'))

        #Flatten Layer for Dens Layer
        model.add(tf.keras.layers.Flatten())

        #Dense Layer with neurons. Use relu activation function
        model.add(tf.keras.layers.Dense(neurons, activation=activation_function_hidden))

    #Output Layer. Also Dense Layer. 10 ouputs (Decimal Numbers)
    model.add(tf.keras.layers.Dense(outputs, activation=activation_function_output))
    
    #Add optimizer and loss function for training
    model.compile(optimizer=optimizer_algo, loss=loss_algo, metrics=metrics_list)

    #Return Model
    return model

def network_train(model, train_x, train_y, epochs: int = 3) -> tf.keras.models.Sequential:
    #Train Model
    model.fit(train_x, train_y, epochs=epochs, verbose=1, batch_size=20, validation_split=0.1, callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
    ])
    return model

def network_test(model, test_x, test_y) -> tf.keras.models.Sequential:
    val_loss, val_acc = model.evaluate(test_x, test_y)
    return val_loss, val_acc

def network_predict(model, predictions, length = None) -> list:
    #Predict
    predictions = model.predict([predictions])
    #Format to usable output
    ret = []
    count = 0
    for p in predictions:
        if length != None and count > length:
            return ret
        ret.append(np.argmax(p))
        count += 1
    return ret

def network_save(model, save_str: str) -> None:
    #Save Model
    model.save(save_str)

def network_load(model_str: str) -> tf.keras.models.Sequential:
    #Load Model
    return tf.keras.models.load_model(model_str)

def get_output_number(dir: str, coin_names: list[str]):
    outputs = 0
    for string in coin_names:
        d = dir + "/" + string
        if os.path.isdir(d) is True:
            dir_length = len(os.listdir(d))
            print(f"Checking {d} ... Length: {dir_length}")
            if dir_length is not None and dir_length != 0:
                outputs += 1
        else:
            print(f"Checking {d} ... NOT FOUND!")
    return outputs

def get_number_from_name(coin_name: str, coin_names_array: list[str]) -> int | None:
    """This function will return the number associated with the name provided.
    """
    count = 0
    for string in coin_names_array:
        if string == coin_name:
            return count
        count += 1
    return None
    
def get_name_from_number(coin_number: int, coin_names_array: list[str]) -> str | None:
    """This function will return the name associated with the number provided.
    """
    if coin_number < len(coin_names_array):
        return coin_names_array[coin_number]
    return None

    
def get_size_of_data(dir: str, coin_names: list[str]):
    count = 0
    for name in coin_names:
        d = dir + "/" + name
        if os.path.isdir(d):
            d_length = len(os.listdir(d))
            if d_length is not None and d_length != 0:
                count += d_length
    return count

def get_images_in_dir(dir: str):
    """This function will return a list of loaded images from dir."""
    images = []
    for i in os.listdir(dir):
        images.append(cv_load_image(dir, i))
    return images

def get_coin_names(dir: str, coin_names: list[str]):
    coin_types_array = []
    for name in coin_names:
        #Get dir and its length
        d = dir + "/" + name
        if os.path.isdir(d):
            d_length = len(os.listdir(d))
            #See if dir is empty
            if d_length is not None and d_length != 0:
                #Add coin type to array
                coin_types_array.append(name)
    return coin_types_array


def get_training_data(dir: str, width: int, height: int, coin_names: list[str]):
    """This function will return a numpy array of images from the training set located in dir
    """
    #Get size of training set data
    d_size = get_size_of_data(dir, coin_names)

    #Create numpy arrays
    image_array = np.empty(shape=(d_size, width, height))
    name_array = np.empty(shape=(d_size))

    #Create array for names of coins
    coin_types_array = []

    #Iterate over all folders of coins
    counter = 0
    for name in coin_names:
        
        #Get dir and its length
        d = dir + "/" + name
        if os.path.isdir(d) is False:
            continue
        d_length = len(os.listdir(d))

        #If dir is not empty, load the images from folder
        if d_length is not None and d_length != 0:

            #Get images
            train_images = get_images_in_dir(d)
            
            #If image number valid, add to array
            if train_images != None:

                #Loop over images in train_images and convert to gray and add to array
                print(f"NAME: {name}: AMOUNT: {len(train_images)} images...")

                #Add coin type to array
                coin_types_array.append(name)

                #Get Name
                train_name = get_number_from_name(name, coin_types_array)

                #Add images to numpy array
                for i in train_images:
                    if i is not None and train_name is not None:
                        train_image_gray = cv_image_to_gray(i)
                        image_array[counter] = train_image_gray
                        name_array[counter] = train_name
                        counter += 1

    print(f"\nLoaded {counter} Images!")
    print(f"Image Types: {coin_types_array}\n")
    return (image_array, name_array, coin_types_array)
