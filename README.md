# Image-Processing-Ex1  
Exercise 1 in the course Image Processing with python  
![Python version](https://img.shields.io/badge/Python-3.8-green)  
![Platform Pycharm](https://img.shields.io/badge/Platform-Pycharm-brightgreen)  
  
ex_utils.py functions:  
   imReadAndConvert: Simple function that get the filename and representation - 1 for Gray and 2 for RGB, read image with OpenCv, and normalize the image data to float.  
   transformRGB2YIQ: Transform a image to YIQ image, with multiplication matrices.  
   transformYIQ2RGB: Transforma image back from YIQ to RGB.  
   hsitogramEqualize: Equalizes image, use Histogram and CumSum.  
   quantizeImage: Make the image with less color. The function get the number of color we want to turn the image into and the number of iteration. Return vector of images that contain each image in iteration and also a vector of the error of each iteration.  
     
gamma.py functions:  
  gammaDisplay: Make the image darker or ligher. Display image with track bar. The track bar represent the Gamma in the formula: (image * A) ^ Gamma
    
ex1_main a test of all functions on the image "beach.jpg"  
