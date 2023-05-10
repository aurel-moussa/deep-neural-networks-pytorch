#Constructing a simple dataset with Pytorch and applying transformations to it

#LIBRARIES IMPORT
import torch
from torch.utils.data import Dataset
torch.manual_seed(1) #manually setting the seed to get the same results every time for testing

#OUR OWN DATASET CLASS (dervied from torch's Dataset class)

# Define class for dataset

class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length #we want a dataset with 100 samples
        self.x = 2 * torch.ones(length, 2) #the first column we call x, and we want it to be (1,1)
        self.y = torch.ones(length, 1) #the second column we call y, and we want it to be 1
        self.transform = transform
     
    # Getter
    def __getitem__(self, index): #defining our own function for indexing elements within the dataset. All we want is the index value
        sample = self.x[index], self.y[index] #define sample to be output as the x(index) for the first output, and then y(index) as the second output
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len
      
# Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.
our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset)) 

# Use loop to print out first 3 elements in dataset
for i in range(3):
    x, y=our_dataset[i]
    print("index: ", i, '; x:', x, '; y:', y)
    
#since our dataset is an iterable object, we can apply loop directly on the dataset object
for x,y in our_dataset:
    print(' x:', x, 'y:', y)

#TRANSFORMATIONS
#Create a transformation classcreate a class for transforming the data.
#In this case, we will try to add 1 to x and multiply y by 2:
#Creating a class for transformations is done to have a single, maintanable object which is well-organized and reusable

# Build the tranform class add_multiply

class add_multiply(object):
    
    # Constructor
    def __init__(self, addx = 1, muly = 2): #this class will add 1 (to x) and multiply (y) by 2
        self.addx = addx
        self.muly = muly
    
    # Executor
    def __call__(self, sample): #if you call this transformation class, you need to input which sample you want to transform it on
        x = sample[0] #x will be the first element of the sample (since that is how we defined it in our dataset)
        y = sample[1] #y will be the second element of the sample (since that is how we defined it in our dataset)
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample

#Create an instance (object) of this class
# Create an add_mult transform object, and an toy_set object

a_m = add_multiply()
data_set = toy_set()

# Use loop to print out first 10 elements in dataset, before and after transformation
for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

#instead of applying the transform after the data_set has been constructed, we can also directly apply it when the dataset is being constructed
#Remember, we have the constructor in toy_set class with the parameter transform = None.
#When we create a new object using the constructor, we can assign the transform object to the parameter transform, 
#as the following code demonstrates.


# Create a new data_set object with add_mult object as transform
cust_data_set = toy_set(transform = a_m)

#Comparing:
# Use loop to print out first 10 elements in dataset
for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

#Squaring and cubing transform class
class squared_cubed(object):
    
    # Constructor
    def __init__(self, squarex = 2, cubey = 3):
        self.squarex = squarex
        self.cubey = cubey
    
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x ** self.squarex
        y = y ** self.cubey
        sample = x, y
        return sample
# Type your code here.

squaring_cubing = squared_cubed()
data_set_2 = toy_set(transform=squaring_cubing)

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = data_set_2[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    
#COMPOSITION OF MULTIPLE TRANSFORMATIONS
#We can do multiple transformations as well

#EXTRA PACKAGGES

# Run the command below when you do not have torchvision installed
# !mamba install -y torchvision
from torchvision import transforms

#Creating another transformation class
# Create tranform class mult

class mult(object):
    
    # Constructor
    def __init__(self, mult = 100):
        self.mult = mult
        
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample
      
# Combine the add_multiply() and mult() classes
data_transform_super_duper = transforms.Compose([add_multiply(), mult()])
print("The combination of transforms (Compose): ", data_transform_super_duper)

#This new super_duper composition will take input, then do add_multiply, then do mult, then output

data_transform_super_duper(data_set[0])

x,y=data_set[0]
x_,y_=data_transform_super_duper(data_set[0])
print( 'Original x: ', x, 'Original y: ', y)
print( 'Transformed x_:', x_, 'Transformed y_:', y_)

#We can also do the transofrm directly when we create the dataset
# Create a new toy_set object with compose object as transform
compose_data_set = toy_set(transform = data_transform_super_duper)

# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)
    
    
