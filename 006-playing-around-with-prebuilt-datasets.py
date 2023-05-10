#INSTALLATION AND LIBRARIES

#if not already installed: !pip install torchvision==0.9.1 torch==1.8.1 
import torch 
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)

#Helper function for displaying images
# Show data by diagram

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))
    
# Run the command below when you do not have torchvision installed
# !mamba install -y torchvision

import torchvision.transforms as transforms
import torchvision.datasets as dsets    

# Import the prebuilt dataset into variable dataset
#We are taking the MNIST dataset which is hand-written digits
dataset = dsets.MNIST( 
    root = './data',  
    download = True, 
    transform = transforms.ToTensor()
)

# Examine whether the elements in dataset MNIST are tuples, and what is in the tuple?
print("Type of the first element: ", type(dataset[0]))
print("The length of the tuple: ", len(dataset[0]))
print("The shape of the first element in the tuple: ", dataset[0][0].shape)
print("The type of the first element in the tuple", type(dataset[0][0]))
print("The second element in the tuple: ", dataset[0][1])
print("The type of the second element in the tuple: ", type(dataset[0][1]))

#What does the first picture look like?
show_data(dataset[0])

#We can have multiple transforms composed together
# Combine two transforms: crop and convert to tensor. Apply the compose to MNIST dataset before dowloading it

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', download = True, transform = croptensor_data_transform)
print("The shape of the first element in the first tuple: ", dataset[0][0].shape)

# Plot the first element in the dataset, it will be cropped
show_data(dataset[0],shape = (20, 20))

#Again, we can do some random flips, because your model off later will not always have perfect pictures
# Construct the compose. Apply it on MNIST dataset. Plot the image out.

fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1),transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', download = True, transform = fliptensor_data_transform)
show_data(dataset[1])
