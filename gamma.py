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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import ex1_utils as utils
import numpy as np
alpha_slider_max = 100
title_window = 'Linear Blend'
myImg = None
A = 1  # the A from power-law expression
gammaNor = 100

def on_trackbar(val):  # Call when the trackbar change, val is the gamma
    #cv2.imshow(title_window, (myImg * A) ** (1 / (val / gammaNor)))
    cv2.imshow(title_window, (myImg * A) ** (val / gammaNor))  # Show the image with the formula

    pass

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    img = utils.imReadAndConvert(img_path, rep)
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    global myImg
    myImg = img
    cv2.namedWindow(title_window)
    cv2.createTrackbar("Gamma x100", title_window, 0, 200, on_trackbar)  # 0-200 represent 0-2
    cv2.setTrackbarPos("Gamma x100", title_window, 100)
    on_trackbar(100)  # set the bar to 1
    cv2.waitKey()

    pass


def main():
    gammaDisplay('beach.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
