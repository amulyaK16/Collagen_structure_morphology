import os
import numpy as np 
from skimage import exposure, restoration, img_as_ubyte
from skimage.util import img_as_float
from numpy.fft import fft2, ifft2
import cv2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import pyfeats


image=Image.open("data/Snap-1188_SHG_mouseLunhFresh.jpg")
#image=Image.open("Experiment-745_good.jpg")

#show image
#image.show()

#converting to grayscale
#L stands for grayscale (luminance)
im_gray=image.convert("L")
#im_gray.show()
#example of showing the image properties
#width, height=image.size
#print("Image size:", width, "x", height)

#convert the image to numpy array
image_array=np.array(im_gray)
#Perform intensity adjustment using the exposure module
im_adjust=exposure.rescale_intensity(image_array, in_range='image', out_range=np.uint8)

#Convert the adjusted numpy array back to PIL image
im_adjust_pil= Image.fromarray(im_adjust)
#show the adjusted image
#im_adjust_pil.show()

#Denoising
#wienerFiltIm=restoration.wiener(image_array, 5, 5) #error
#http://scipy-lectures.org/advanced/image_processing/?fbclid=IwAR3rQdRZ0GBNGsIVFR490lPlYXpjxeFYvpX3pZ2FR4HpXa5nFmA_BYymD0E#denoising
#https://stackoverflow.com/questions/53883717/applying-wiener-filter-to-remove-noise-using-python


# Apply Wiener Filter
#kernel = gaussian_kernel(5)
#filtered_img = wiener_filter(im_adjust, kernel, K = 5)
#filtered_img_pil=Image.fromarray(filtered_img)
#filtered_img_pil.show()

#Apply median filter
median= im_adjust_pil.filter(ImageFilter.ModeFilter(size = 3)) 
median.show()
median_array=np.array(median)
#FOS and SOS
# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
#https://realpython.com/image-processing-with-the-python-pillow-library/
#mask = median.filter(ImageFilter.FIND_EDGES)
#mask.show()
#mask_array=np.array(mask)
#features = {}
#features['A_FOS'] = pyfeats.fos(median_array, mask_array)
#print(features)
#First Order Statistics
#create own mask using edge detection and binary mask
#https://github.com/giakou4/pyfeats

#get histogram (CREATE HISTOGRAM Properly, make it for a specific roi)
hist,bin=np.histogram(median_array.ravel(),256, [0,255])
plt.xlim([0,255])
plt.plot(hist)
plt.title('histogram of filtered image')

plt.show()

#calculate features
features=np.zeros(4, np.double)
#set bin range
bins=(255-0) + 1
#np.arange returns evenly spaced values within a given interval
i=np.arange(0,bins) 

#Mean
features[0]=np.dot(i,hist)

#Standard Deviation
features[1]=np.sqrt(sum(np.multiply(((i-features[0])**2),hist)))
#Kurtosis
features[2]=sum(np.multiply(((i-features[0])/features[1])**4,hist))
#Skew
features[3]=sum(np.multiply(((i-features[0])/features[1])**3,hist))
print("mean, std dev, kurtosis, skewness")
print("-------------------------------")
print(features)
#close image
image.close()