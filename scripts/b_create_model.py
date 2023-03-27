from neural_network import *
import tensorflow as tf

#Neural Network Constants
HIDDEN_LAYERS = 6
NEURONS = 1024
ACTIVATION_FUNCTION_HIDDEN_LAYER = tf.nn.relu
ACTIVATION_FUNCTION_OUTPUT_LAYER = tf.nn.softmax
OPTIMIZER_ALGORITHM = 'adam'
LOSS_ALGORITHM = 'sparse_categorical_crossentropy'
METRICS_LIST = ['accuracy']

#Data Constants
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
TRAIN_DIR = "./neural_network/train_data"
MODEL_DIR = "./neural_network/models"
MODEL_NAME = "coin_counter.model"
COIN_TYPES = ['penny', 'nickel', 'dime', 'quarter', 'half_dollar', 'dollar_coin']
COIN_VALUES= [   0.01,     0.05,   0.10,      0.25,          0.50,          1.00]

#Get Training Data
x_train, y_train, z_train = get_training_data(TRAIN_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, COIN_TYPES)
x_train_n, y_train_n, z_train_n = cv_image_normalize(x_train), y_train, z_train

#Get the amount of outputs from training data
print("\nDetermining Outputs...\n")
outputs = get_output_number(TRAIN_DIR, COIN_TYPES)
print(f"\n{outputs} Ouputs.")

#Create the network
print("\nCreating Network...\n")
model = network_create(outputs=outputs, 
                        hidden_layers=HIDDEN_LAYERS, 
                        neurons=NEURONS, 
                        activation_function_hidden=ACTIVATION_FUNCTION_HIDDEN_LAYER,
                        activation_function_output=ACTIVATION_FUNCTION_OUTPUT_LAYER,
                        optimizer_algo=OPTIMIZER_ALGORITHM,
                        loss_algo=LOSS_ALGORITHM,
                        metrics_list=METRICS_LIST)

#Training Once to set input sizes
network_train(model, x_train_n, y_train_n, 1)

#Save the network
print("\nSaving Model...\n")
network_save(model, f"{MODEL_DIR}/{MODEL_NAME}")
print("\nModel Created!\n")
print(f"Saved at: {MODEL_DIR}/{MODEL_NAME}\n")

