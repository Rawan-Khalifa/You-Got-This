#!/usr/bin/env python
# coding: utf-8

# In[2]:


#loading the data files
import numpy as np
data = np.load('chestmnist.npz')
print(data.files)


# In[3]:

#Prompt [1]

import matplotlib.pyplot as plt

image = data['train_images'][0]

# Create a figure and axis
fig, ax = plt.subplots()

# Display the image on the axis
ax.imshow(image, cmap='gray')

# Show the plot
plt.show()


# In[4]:

#Prompt [2]

# Define the filter 
#subtracts each pixel value from the maximum possible pixel value 
filter = np.full_like(image, 255) - image

# Create a figure and axis
fig, ax = plt.subplots()

# Display the filtered image on the axis
ax.imshow(filter, cmap='gray')

# Show the plot
plt.show()


# In[5]:

#Prompt [3]

# adapted from "https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43"
#this contrast enhancement filter to improve the contrast by redistributing pixel intensities

from skimage import exposure, io


# Apply histogram equalization
equalized = exposure.equalize_hist(image)

# Display the original and equalized images
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
ax1.imshow(image, cmap='gray')
ax2.imshow(equalized, cmap='gray')
plt.show()

