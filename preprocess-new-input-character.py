import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Converts a colour m x n x 3 image into an m x n grayscale image
def rgb2gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

char = mpimg.imread('char.png')     
char_gray = rgb2gray(char)    

# Finds the maximum difference between entries in a 2 x 2 matrix
def maxdiff(mat):
    arr = mat.flatten()
    min_entry = arr[0]
    max_entry = arr[0]
    for i in range(1,len(arr)):
        min_entry = min(min_entry, arr[i])
        max_entry = max(max_entry, arr[i])
    return max_entry-min_entry

# Converts the grayscale character into a 10 x 10 matrix of 0's and 1's
char_mat = np.empty([10,10])
h = char_gray.shape[0]//10
v = char_gray.shape[1]//10
for i in range(10):
  for j in range(10):
    if maxdiff(char_gray[i*h : (i+1)*h, j*v : (j+1)*v]) > 0.1:
      char_mat[i,j] = 1
    else:
      char_mat[i,j] = 0
