import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image

def load_img(path_to_img):
    '''
    Loads an image file into an array.

    Keyword Arguments:
    path_to_img - Path to the image we wish to load

    Returns:
    A numpy array containing our image information.
    '''
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension 
    img = np.expand_dims(img, axis=0)
    return img

def imshow(img, title=None):
    '''
    Shows a desired image in matplotlib.

    Keyword Argments:
    img   - A numpy array containing our image.
    title - The desired title of graph.
    '''
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display 
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)

def load_and_process_img(path_to_img):
    '''
    Loads an image into memory, and preprocesses in Keras.

    Keyword Arguments:
    path_to_img - File location of desired image.

    Returns:
    Processed image.
    '''
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    '''
    Reverses the steps of a processed image in Keras.
    '''
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x