#Some of the techniques listed herein are also applicable to any other big boy datasets
#l build a dataset objects for image ( many of the processes can be applied to a larger dataset).
#Then you will apply pre-build transforms from Torchvision Transforms to that dataset.

#PREPARATION
#Download the datasets in your Terminal/Console and unzip them
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/labs/Week1/data/img.tar.gz -P /resources/data
!tar -xf /resources/data/img.tar.gz 

# LIBRARIRES
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0) #forcing manual seed for testing
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os

#AUXILIARY HELPER NICE BOY FUNCTIONS
#Showing image in smaller shape and the y title
def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])
    
directory=""
csv_file ='index.csv'
csv_path=os.path.join(directory,csv_file)

# load the CSV file and convert it into a datafram
#using the Pandas function read_csv()
#vie the first 10 items in the dataframe using the method head

data_name = pd.read_csv(csv_path)
data_name.head()

#The first column of the dataframe corresponds to the type of clothing. 
#The second column is the name of the image file corresponding to the clothing. 
#You can obtain the path of the first file by using the method  DATAFRAME.iloc[0, 1]. 
#The first argument corresponds to the sample number, and the second input corresponds to the column index. 

#Python iloc() function enables us to select a particular cell of the dataset, that is, it helps us select a value that belongs 
#to a particular row or column from a set of values of a data frame or dataset.
#With iloc() function, we can retrieve a particular value belonging to a row and column using the index values assigned to it.
#Remember, iloc() function accepts only integer type values as the index values for the values to be accessed and displayed.

# Get the value on location row 0, column 1 (Notice that index starts at 0)
#rember this dataset has only 100 samples to make the download faster  
print('File name:', data_name.iloc[0, 1])

# Get the value on location row 0, column 0 (Notice that index starts at 0.)
print('y:', data_name.iloc[0, 0])

# Print out the file name and the class number of the element on row 1 (the second row)
print('File name:', data_name.iloc[1, 1])
print('class or y:', data_name.iloc[1, 0])

print('The number of rows: ', data_name.shape[0])

# Combine the directory path with file name
image_name =data_name.iloc[1, 1]
image_name

#find the full image path
image_path=os.path.join(directory,image_name)
image_path

#use the function Image.open to store the image to the variable image and display the image and class .
image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[1, 0])
plt.show()

#If you are interested in sample number 20 instead
# Plot the 20th image

image_name = data_name.iloc[19, 1]
image_path=os.path.join(directory,image_name)
image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[19, 0])
plt.show()

#CREATE A DATASET CLASS FOR IMAGES USING THE ABOVE MAGIC
# Create your own dataset object

class AmazingImageDataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        
        # Image directory
        self.data_dir=data_dir
        
        # The transform is going to be used on image
        self.transform = transform
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        # Load the CSV file that contain image info
        self.data_name= pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx): #idx to specify the sample oyu want
        
        # Image file path
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx, 1])
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 0]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
      
# Create the dataset object
dataset_example = AmazingImageDataset(csv_file=csv_file, data_dir=directory)

#Lets have a look at the first sample
image=dataset_example[0][0]
y=dataset_example[0][1]
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show()

#Lets look at 10th sample
image=dataset[9][0]
y=dataset[9][1]
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show()

#IMAGE TRANSFORMATIONS
import torchvision.transforms as transforms #pre-made transofmrations

#crop, then transform totensor
croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset_example_2 = AmazingImageDataset(csv_file=csv_file , data_dir=directory,transform=croptensor_data_transform )
print("The shape of the first element tensor: ", dataset_example_2[0][0].shape)

# Plot the first element in the dataset
show_data(dataset_example_2[0],shape = (20, 20)) #we see there is less to see!
# Plot the second element in the dataset
show_data(dataset_example_2[1],shape = (20, 20))

#Flip randomly, then transform to tensor
fliptensor_data_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1),transforms.ToTensor()]) 
#What is the purpose of random flip? Well, images which a camera makes will not alwaysbe the right way up.

#So you need to train your model on images that are not perfectly aligned, because that is what it will experience later in real-life
dataset_example_3 = AmazingImageDataset(csv_file=csv_file , data_dir=directory,transform=fliptensor_data_transform )
show_data(dataset_example_3[1])
