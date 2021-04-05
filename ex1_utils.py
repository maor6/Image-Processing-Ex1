"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
# save the YIQ kernel
YIQ_kernel = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])

import cv2
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 205783350


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # check if we want gray or color image, and normalize the image with cv2 function
    img = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # open with cv2 need to transform BGR to RGB

    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)

    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    img = imReadAndConvert(filename, representation)
    if representation == LOAD_GRAY_SCALE:
        plt.imshow(img, cmap="gray")  # display gray image
    else:
        plt.imshow(img)

    plt.show()
    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # normalize the image to float and then multiply with the YIQ kernel matrix
    # convert to shape we can multiply and then back to normal shape
    shapeToMul = (-1, 3)
    imgRGB = cv2.normalize(imgRGB, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    realShap = imgRGB.shape
    YIQ = np.array(imgRGB.reshape(shapeToMul) @ (YIQ_kernel.transpose())).reshape(realShap)

    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # multiply with the Reverse YIQ kernel matrix
    # convert to shape we can multiply and then back to normal shape
    shapeToMul = (-1, 3)
    realShap = imgYIQ.shape
    RGB = np.array(imgYIQ.reshape(shapeToMul) @ (np.linalg.inv(YIQ_kernel).transpose())).reshape(realShap)

    return RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    numToNorm = 255
    realShape = np.array(imgOrig).shape
    RGBSize = 3
    image = imgOrig.copy()
    if len(realShape) == RGBSize:  # transform to YIQ and take the Y channel
        YIQImage = transformRGB2YIQ(image)
        image = YIQImage[:, :, 0]

    image = (image * numToNorm).astype('uint32')  # normalize to int before we calculate histogram
    flatImg = image.flatten()
    histOrg, bins = np.histogram(flatImg, bins=256, range=[0, 255])
    cumSumOrg = histOrg.cumsum()
    LUT = numToNorm * (cumSumOrg / cumSumOrg.max())
    imEq = LUT[flatImg].reshape(realShape[0], realShape[1])  # get the new image from the look up table
    histEQ, new_bins = np.histogram(imEq, bins=256, range=[0, 255])
    if len(realShape) == RGBSize:  # transform back to RGB and get all the channels YIQ
        imEq = imEq / numToNorm
        YIQImage[:, :, 0] = imEq
        imEq = transformYIQ2RGB(YIQImage)
        imEq = imEq.astype('float64')

    return imEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    numToNorm = 255
    upperBound = 256
    RGBSize = 3
    RGB = False
    realShape = imOrig.shape
    Y = imOrig.copy()
    if len(realShape) == RGBSize:
        YIQ = transformRGB2YIQ(imOrig)
        Y = YIQ[:, :, 0]
        RGB = True

    Y = (Y * numToNorm).astype('uint32')
    hisOrg, bins = np.histogram(Y, bins=256, range=[0, 255])
    z = [(i * upperBound) // nQuant for i in range(nQuant + 1)]  # make boundaries
    q = meanIn(z, hisOrg)
    error = []
    quaImg = []

    for i in range(nIter):
        z[1:len(z) - 1] = (np.add(q[:len(q) - 1], (q[1:]))) // 2
        q = meanIn(z, hisOrg)
        err = 0
        for k in range(len(z) - 1):
            for j in range(z[k], z[k + 1]):
                err += ((q[k] - j) ** 2) * hisOrg[j]  # over on z and calculate the error

        error.append(np.min(err) / np.size(Y))  # save all the errors
        LUT = []
        for m in range(len(z) - 1):
            for x in range(z[m], z[m + 1]):
                LUT.append(q[m])

        Ytemp = np.array(LUT)[Y].reshape(realShape[0], realShape[1]) / numToNorm
        if RGB:  # transform back to RGB and get all the channels YIQ
            YIQtemp = YIQ.copy()
            YIQtemp[:, :, 0] = Ytemp
            YIQtemp = transformYIQ2RGB(YIQtemp)
            Ytemp = YIQtemp

        quaImg.append(Ytemp)

    return quaImg, error


"""
    calculate the color of each boundaries
    :Param z: vector of boundaries
    :Param hisOrg: vector of original image histogram 
    :Return vector of color
    
"""
def meanIn(z: np.ndarray, hisOrg: np.ndarray) -> np.array:
    q = []
    for i in range(len(z) - 1):
        q.append((hisOrg[z[i]:z[i + 1]] * range(z[i], z[i + 1])).sum() // hisOrg[z[i]:z[i + 1]].sum())

    return q
