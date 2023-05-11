from neural_network import *
import tensorflow as tf
import time

#Training Constants
EPOCH_CHUNKS = 50
ACCURACY_TARGET = 0.99
MAX_TESTS = 100
MAX_TIME = None

#Data Constants
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
TRAIN_DIR = "./neural_network/train_data"
TEST_DIR = "./neural_network/test_data"
MODEL_DIR = "./neural_network/models"
MODEL_NAME = "coin_counter.model"
COIN_TYPES = ['penny', 'nickel', 'dime', 'quarter', 'half_dollar', 'dollar_coin']
COIN_VALUES= [   0.01,     0.05,   0.10,      0.25,          0.50,          1.00]

def main():
    #Get the amount of outputs from training data
    outputs = get_output_number(TRAIN_DIR, COIN_TYPES)

    #Get Training Data
    x_train, y_train, z_train = get_training_data(TRAIN_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, COIN_TYPES)
    x_train_n, y_train_n, z_train_n = cv_image_normalize(x_train), y_train, z_train

    #Get Test Data
    x_test, y_test, z_test = get_training_data(TEST_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, COIN_TYPES)
    x_test_n, y_test_n, z_test_n = cv_image_normalize(x_test), y_test, z_test

    #Create the network
    model = network_load(f"{MODEL_DIR}/{MODEL_NAME}")

    #Train the network
    finished = False
    test_count = 0
    start_time = time.time()
    while finished is False:

        #Train Network
        print("\nTraining Network... [Exit: Ctrl + C]")
        network_train(model, x_train_n, y_train_n, EPOCH_CHUNKS)

        #Save Network
        print("\nTraining Finished. Saving... [Do Not Exit!]")
        network_save(model, f"{MODEL_DIR}/{MODEL_NAME}")

        #Test Network
        print("\nSaving Finished. Testing Network... [Exit: Ctrl + C]")
        test_count += 1
        current_time = time.time()
        val_loss, val_acc = network_test(model, x_test_n, y_test_n)

        #Print Results
        test_str = f"Test ({test_count} / {MAX_TESTS})"
        loss_str = f"Loss: {val_loss}"
        acc_str = f"Accuracy: ({round(val_acc*100, 2)} % / {round(ACCURACY_TARGET*100, 2)} %)"
        time_str = f"Time: ({round(current_time - start_time, 2)} / {MAX_TIME}) second(s)"
        print(f"\n{test_str}\n{loss_str}\n{acc_str}\n{time_str}")

        #Exit Training
        if ACCURACY_TARGET is not None and val_acc >= ACCURACY_TARGET:
            print(f"\n[Training Finished] Accuracy Target Hit:\n{test_str},\n{acc_str}\n{time_str}")
            finished = True

        elif MAX_TESTS is not None and test_count >= MAX_TESTS:
            print(f"\n[Training Finished] Max Tests Hit:\n{test_str},\n{acc_str}\n{time_str}")
            finished = True

        elif MAX_TIME is not None and current_time - start_time >= MAX_TIME:
            print(f"\n[Training Finished] Max Time Hit:\n{test_str},\n{acc_str}\n{time_str}")
            finished = True

if __name__ == "__main__":
    main()

