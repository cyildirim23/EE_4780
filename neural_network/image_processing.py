import pygame
import cv2 as cv
import numpy as np
import tensorflow as tf
import os

def cv_image_to_surface(cvImage: np.ndarray) -> pygame.Surface:
    """ This Function converts a cv (OpenCV) Image Format into a Pygame Surface.
    """
    if cvImage.dtype.name == 'uint16':
        cvImage = (cvImage / 256).astype('uint8')
    size = cvImage.shape[1::-1]
    if len(cvImage.shape) == 2:
        cvImage = np.repeat(cvImage.reshape(size[1], size[0], 1), 3, axis = 2)
        format = 'RGB'
    else:
        format = 'RGBA' if cvImage.shape[2] == 4 else 'RGB'
        cvImage[:, :, [0, 2]] = cvImage[:, :, [2, 0]]
    surface = pygame.image.frombuffer(cvImage.flatten(), size, format)
    return surface.convert_alpha() if format == 'RGBA' else surface.convert()

def cv_image_to_gray(cvImage: np.ndarray) -> np.ndarray:
    """This function converts a color cv (OpenCV) Image into a Grayscale Image
    """
    img = cvImage
    if len(cvImage.shape) == 3:
        img = cv.cvtColor(cvImage, cv.COLOR_BGR2GRAY)
    return img

def cv_image_resize(cvImage: np.ndarray, width: int = 128, height: int = 128) -> np.ndarray:
    """This function scales a grayscale cv (OpenCV) image to (width, height).
        Default: 128 x 128
    """
    return cv.resize(cvImage, (width, height))

def cv_image_normalize(input_list: np.ndarray) -> np.ndarray:
    """This function normalizes the images in the list for neural networking
    """
    return tf.keras.utils.normalize(input_list, axis=1)

def cv_image_get_circles(cvImage: np.ndarray, 
                         gradient = cv.HOUGH_GRADIENT,
                         dp = 1.15,
                         minDist = 60,
                         param1 = 50,
                         param2 = 45,
                         minRadius = 5,
                         maxRadius = 70

    ) -> np.ndarray:
    """This function uses the Hough Circles function from cv (OpenCV) to get an array of circles in cvImage.
    """
    circles = cv.HoughCircles(cvImage, gradient, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        return np.round(circles[0, :]).astype("int")
    return None

def cv_filter_circles(circles: np.ndarray, small_bias: bool = True):
    return_circles = []
    for (x1, y1, r1) in circles:
        discard_circle = False
        for (x2, y2, r2) in circles:
            if (x1, y1, r1) == (x2, y2, r2):
                continue
            dis = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dis < r1 + r2:
                if small_bias:
                    if r1 > r2:
                        discard_circle = True
                        break
                else:
                    if  r1 < r2:
                        discard_circle = True
                        break
        if not discard_circle:
            return_circles.append((x1, y1, r1))
    return np.array(return_circles)
    
def cv_create_circle_images(cvImage: np.ndarray, 
                            circles: np.ndarray, 
                            width: int = 128, 
                            height: int = 128, 
                            buffer: int = 10, 
                            paused: bool = False
    ) -> np.ndarray:
    """This function creates scaled images from the output of cv (OpenCV) Hough Function.
    """
    #Create Arrays for circles and images
    coin_circles = []
    coin_images_arr = []
    coin_images_py = []

    if circles is None:
        return (None, None, None)

    #Loop over all circles detected by Hough Function
    for (x, y, r) in circles:

        #Get starting position of rectangle
        x1 = x - r - buffer
        y1 = y - r - buffer

        #Get end position of rectangle
        x2 = x1 + r*2 + buffer*2
        y2 = y1 + r*2 + buffer*2

        #If rectangle is valid, get image
        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
            
            #Create sub-image from main image for circle
            res_img = cvImage[y1:y2, x1:x2]
            res_img_scaled = cv_image_resize(res_img, width, height)

            #Convert to Pygame Surface
            py_img2 = cv_image_to_surface(res_img_scaled)

            if res_img_scaled.shape == (128, 128) and paused is False:
                
                #Append circle coordinate to list
                coin_circles.append((x, y, r))

                #Append image to list
                coin_images_py.append(py_img2)
                coin_images_arr.append(res_img_scaled)
    
    #Return the lists
    coin_images_nd = np.array(coin_images_arr)
    return (coin_circles, coin_images_nd, coin_images_py)

def cv_save_image(cvImage: np.ndarray, dir: str, image_name: str):
    """This function saves an image with image_name to the dir directory.
    """
    count = len(os.listdir(dir))
    if count is None:
        count = 0
    d = dir + "/" + image_name + "_" + str(count) + ".PNG"

    #Make sure image does not already exist
    while os.path.isfile(d) is True:
        count += 1
        d = dir + "/" + image_name + "_" + str(count) + ".PNG"

    cv.imwrite(d, cvImage)

def cv_load_image(dir: str, image_name: str, image_number: int = None):
    """This function loads an image from dir with image_name_image_number.PNG as its name.
    """
    d = None
    if image_number is not None:
        d = dir + "/" + image_name + "_" + str(image_number) + ".PNG"
    else:
        d = dir + "/" + image_name
    return cv.imread(d)
