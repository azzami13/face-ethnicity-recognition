import cv2
import numpy as np

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_median_blur(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_canny_edge(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)