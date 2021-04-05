# Image-Processing-Ex1
Exercise 1 in the course Image Processing with python
Python version 3.8
Platform Pycharm

ex_utils.py functions:
  imReadAndConvert: Simple function that read image, and normalize the image data to float.
  transformRGB2YIQ: Transform a image to YIQ image, with multiplication matrices.
  transformYIQ2RGB: Transforma image back from YIQ to RGB.
  hsitogramEqualize: Equalizes image, use Histogram and CumSum.
  quantizeImage: Make the image with less color. The function get the number of color we want to turn the image into and, the number of iteration that we want to improve the quantization
  
gamma.py functions:
  gammaDisplay: Make the image darker or ligher. Display image with track bar. The track bar represent the Gamma in the formula: (image * A) ^ Gamma
  
  
ex1_main a test of all functions on the image "beach.jpg"
