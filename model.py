import argparse
import csv
import cv2
import math
import pickle
import json

import matplotlib.image as mpimg
import numpy as np

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# Angle offset for the left and right cameras.
ANGLE_OFFSET = 0.25

# Angle offsets applied to center, left and right image
ANGLE_OFFSETS = [0.0, ANGLE_OFFSET, -ANGLE_OFFSET]

# Parse arguments
parser = argparse.ArgumentParser(description='Nvidia based training model.')
parser.add_argument('csv_file',
                    type=str,
                    help='CSV file to use for training.')
parser.add_argument('epochs',
                    type=int,
                    help='Training epochs.')
parser.add_argument('output',
                    type=str,
                    help='Where the model is going to be saved.')
parser.add_argument('fine_tuning',
                    type=int,
                    help='Fine tuning the top fully connected layers.')
parser.add_argument('learning_rate',
                    type=float,
                    help='Learning Rate.')
args = parser.parse_args()


def resize(x):
    """Resize the input image to be 66 x 200. Color is not altered."""

    # Get the image shape
    height = x.shape[0]
    width = x.shape[1]

    # Compute the scaling factor
    factor = 200.0 / float(width)

    # Resize
    resized_size = (int(width*factor), int(height*factor))
    x = cv2.resize(x, resized_size)
    crop_height = resized_size[1] - 66
    return x[crop_height:, :, :]


def load_data(input_csv):
    """ Read the input CSV file.

    Returnes:
        - A list with 2 elements. Element at index 0 is a list of lists of 3 file
        names (i.e. center image, left image, right image). element at index 1
        is a list of steering angles.
    """
    print("Reading CSV...{}".format(input_csv))
    with open(input_csv) as f:
        X = []
        y = []

        csv_reader = csv.reader(f)
        for line in csv_reader:
            # Skip header
            if line[0] == 'center': continue

            streer_angle   = float(line[3])
            img_name       = line[0].strip()
            img_name_left  = line[1].strip()
            img_name_right = line[2].strip()

            X.append([img_name, img_name_left, img_name_right])
            y.append(streer_angle)

    return X, y


def normalize(images):
    """ Normalizes the input between -0.5 and +0.5 """
    return images / 255.0 - 0.5


def define_network(input_shape, fine_tuning=False, learning_rate=0.001):
    """ Here we define the Nvidia network.

        Further reading at:
        https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    weight_init='glorot_uniform'
    padding = 'valid'
    dropout_prob = 0.25

    # Define model
    model = Sequential()

    # Normalize the input without changing shape
    model.add(Lambda(normalize, input_shape=input_shape, output_shape=input_shape))

    # Convolution layers
    model.add(Convolution2D(24, 5, 5,
                            border_mode=padding,
                            init = weight_init, subsample = (2, 2)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5,
                            border_mode=padding,
                            init = weight_init, subsample = (2, 2)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5,
                            border_mode=padding,
                            init = weight_init, subsample = (2, 2)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,
                            border_mode=padding,
                            init = weight_init, subsample = (1, 1)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,
                            border_mode=padding,
                            init = weight_init, subsample = (1, 1)))

    # Fully connected classifier
    model.add(Flatten())
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(100, init = weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(50, init = weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(10, init = weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(1, init = weight_init, name = 'output'))

    # Display the model summary
    model.summary()

    # Compile it
    model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))
    return model


def correct_training_samples(X, batch_size):
    """ Given the number of samples and the batch size, determines the correct
        number of samples to generate to avoid Keras warnings.
    """
    return int(math.ceil(float(X) / float(batch_size)) * batch_size)


def do_augmentation(image, steering):
    """ Performs image augmentation (horizzontal shift and vertical flip). """

    # Maximum shift of the image, in pixels
    trans_range = 50

    # Compute translation and corresponding steering angle
    tr_x = np.random.uniform(-trans_range, trans_range)
    steering = steering + (tr_x / trans_range) * ANGLE_OFFSET

    # Get the image shape
    rows = image.shape[0]
    cols = image.shape[1]

    # Warp the image using openCV
    t_matrix = np.float32([[1,0,tr_x],[0,1,0]])
    img = cv2.warpAffine(image, t_matrix, (cols, rows))

    # Determines if image is going to be flipped
    flip = np.random.randint(2)

    # If needed, flip the image
    if flip:
        image = cv2.flip(image, 1)
        steering = -steering

    return image, steering


def training_generator(X, y, batch_size):
    """ Train the model in batches.
        We cannot guarantee to load all the training images in memory (RAM)
        therefore, we need a generator which reads the images from disk and
        perform augmentation on the fly.

        The skelethon of this method is inspired by:

        https://keras.io/models/sequential/#fit_generator
    """
    # Supply training images indefinitely
    while 1:
        # Declare output data
        x_out = []
        y_out = []

        for i in range(0, batch_size):
            # Get random index to an element in the dataset.
            idx = np.random.randint(len(y))

            # Randomly select which of the 3 images (center, left, right) to use
            idx_img = np.random.randint(len(ANGLE_OFFSETS))

            # Read image and steering angle (with added offset)
            x_i = mpimg.imread(X[idx][idx_img].strip())
            y_i = y[idx] + ANGLE_OFFSETS[idx_img]

            # Preprocess image
            x_i = resize(x_i)

            # Augment data
            x_i, y_i = do_augmentation(x_i, y_i)

            # Add to batch
            x_out.append(x_i)
            y_out.append(y_i)

        yield (np.array(x_out), np.array(y_out))


def validation_generator(X, y):
    """ Provides images for validation. Without augmentation. """
    # Validation generator
    while 1:
        for i in range(len(y)):
            # Read image and steering angle
            x_out = mpimg.imread(X[i][0])
            y_out = np.array([[y[i]]])

            # Preprocess image
            x_out = resize(x_out)
            x_out = x_out[None, :, :, :]

            # Return the data
            yield x_out, y_out


def train_network(model, epochs, X_train, y_train):
    """ A wrapper to train the network. """
    print("Training...")

    batch_size = 64
    train_samples = 5 * correct_training_samples(len(y_train), batch_size)
    valid_samples = len(y_train)

    train_gen = training_generator(X_train, y_train, batch_size)
    valid_gen = validation_generator(X_train, y_train)

    # We are exploiting the parallelism of Nvidia GPUs with nb_worker > 1.
    model.fit_generator(generator=train_gen,
                        samples_per_epoch=train_samples,
                        validation_data=valid_gen,
                        nb_val_samples=valid_samples,
                        nb_epoch=epochs,
                        nb_worker=8,
                        pickle_safe=True,
                        verbose=1)


def save_network(out_file, model):
    """ Saves model (json) and weights (h5) to disk. """
    print("Saving model...")

    model.save_weights(out_file + ".h5")

    model_json = model.to_json()
    with open(out_file + ".json", "w") as f:
         json.dump(model_json, f)

    print("Saved")

def load_trained_network(in_file, learning_rate=0.001):
    """ Load a previous trained network for fine tuning.

        Note:
        Instead of re-training the entire network, from a previous saved state,
        it would be better to just re-train the fully connected layers.
        However, I faced issues on Keras when exporting the partially trained
        network (Keras exported only the re-trained layer Instead of the entire
        network). In particular, it looks like that Keras's sequential APIs have
        some troubles when dealing with partially re-trained models.
        A solution would be using Keras's functional APIS.
    """
    model = None
    with open(in_file + '.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr = learning_rate))
        model.load_weights(in_file + ".h5")
    return model


def main():
    """ Main routine. """

    # Load the driving_log.csv file
    X, y = load_data(args.csv_file)
    fine_tuning = (args.fine_tuning == 1)

    # Determine if we are fine tuning a previous trained network.
    if fine_tuning:
        model = load_trained_network(args.output, args.learning_rate)
    else:
        model = define_network((66, 200, 3), fine_tuning, args.learning_rate)

    # Training
    train_network(model, args.epochs, X, y)

    # Save the model with a different name if it was fine tuned.
    if fine_tuning:
        save_network(args.output + '_fine_tuned', model)
    else:
        save_network(args.output, model)

    print("Done")

if __name__ == '__main__':
    main()
