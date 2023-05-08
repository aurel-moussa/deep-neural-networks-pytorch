
# Basics of tensor operations

# Preparation
# Library Imports

import torch 
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline  

# Check version that is installed
torch.__version__

# Helper function for plotting diagrams
# Plot vecotrs, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]
    
def plotVec(vectors):
    ax = plt.axes()
    
    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width = 0.05,color = vec["color"], head_length = 0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])
    
    plt.ylim(-2,2)
    plt.xlim(-2,2)

# Tensor Type and Tensor Shape
# Convert a integer list with length 5 to a tensor

ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])

#Specific tensor types
print("The dtype of tensor object after converting it to tensor: ", ints_to_tensor.dtype)
print("The type of tensor object after converting it to tensor: ", ints_to_tensor.type())

#Type of object according to Phython
type(ints_to_tensor)

# Convert a float list with length 5 to a tensor

floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
print("The dtype of tensor object after converting it to tensor: ", floats_to_tensor.dtype)
print("The type of tensor object after converting it to tensor: ", floats_to_tensor.type())

# Specify the type of float you want when creating tensor
list_floats=[0.0, 1.0, 2.0, 3.0, 4.0]
floats_int_tensor=torch.tensor(list_floats,dtype=torch.int64)

print("The dtype of tensor object is: ", floats_int_tensor.dtype)
print("The type of tensor object is: ", floats_int_tensor.type())

#Change the type of tensor of an existing tensor
# Convert a integer list with length 5 to float tensor

new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
new_float_tensor.type()
print("The type of the new_float_tensor:", new_float_tensor.type())

# Another method to convert the integer list to float tensor

old_int_tensor = torch.tensor([0, 1, 2, 3, 4])
new_float_tensor = old_int_tensor.type(torch.FloatTensor)
print("The type of the new_float_tensor:", new_float_tensor.type())

#The tensor_obj.size() helps you to find out the size of the tensor_obj. The tensor_obj.ndimension() shows the dimension of the tensor object.

# Introduce the tensor_obj.size() & tensor_ndimension.size() methods

print("The size of the new_float_tensor: ", new_float_tensor.size())
print("The dimension of the new_float_tensor: ",new_float_tensor.ndimension())

#Reshaping tensors

#he tensor_obj.view(row, column) is used for reshaping a tensor object.
# What if you have a tensor object with torch.Size([5]) as a new_float_tensor as shown in the previous example?
# After you execute new_float_tensor.view(5, 1), the size of new_float_tensor will be torch.Size([5, 1]).
#This means that the tensor object new_float_tensor has been reshaped from a one-dimensional tensor object with 5 elements to a two-dimensional tensor object with 5 rows and 1 column.

twoD_float_tensor = new_float_tensor.view(5, 1)
print("Original Size: ", new_float_tensor)
print("Size after view method", twoD_float_tensor)

#What if you have a tensor with dynamic size but you want to reshape it? You can use -1 to do just that.
# Introduce the use of -1 in tensor_obj.view(row, column) method

twoD_float_tensor = new_float_tensor.view(-1, 1)
print("Original Size: ", new_float_tensor)
print("Size after view method", twoD_float_tensor)

#You get the same result as the previous example. The -1 can represent any size. However, be careful because you can set only one argument as -1

#Converting other objects into and from Tensors
#Converting Numpy array to Tensor


numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)

print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())

#Convert Tensor to Numpy

back_to_numpy = new_tensor.numpy()
print("The numpy array from tensor: ", back_to_numpy)
print("The dtype of numpy array: ", back_to_numpy.dtype)

# Set all elements in numpy array to zero 
numpy_array[:] = 0
print("The new tensor points to numpy_array : ", new_tensor)
print("and back to numpy array points to the tensor: ", back_to_numpy)

#Converting Panda series to Tensor

pandas_series=pd.Series([0.1, 2, 0.3, 10.1])
new_tensor=torch.from_numpy(pandas_series.values) #Actually uses the values from panda series to the from_numpy function
print("The new tensor from numpy array: ", new_tensor)
print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())

#Selecting values within the tensor
this_tensor=torch.tensor([0,1, 2,3]) 

#Item gives the value saved inside the tensor. Item can only be called on a single item.

print("the first item is given by",this_tensor[0].item(),"the first tensor value is given by ",this_tensor[0])
print("the second item is given by",this_tensor[1].item(),"the second tensor value is given by ",this_tensor[1])
print("the third  item is given by",this_tensor[2].item(),"the third tensor value is given by ",this_tensor[2])

#List is to list all the values inside the tensor
torch_to_list=this_tensor.tolist()

print('tensor:', this_tensor,"\nlist:",torch_to_list)

#Indexing
index_tensor = torch.tensor([0, 1, 2, 3, 4])
print("The value on index 0:",index_tensor[0])
print("The value on index 1:",index_tensor[1])
print("The value on index 2:",index_tensor[2])
print("The value on index 3:",index_tensor[3])
print("The value on index 4:",index_tensor[4])

#Changing values via indexing
tensor_sample = torch.tensor([20, 1, 2, 3, 4])
# Change the value on the index 0 to 100

print("Inital value on index 0:", tensor_sample[0])
tensor_sample[0] = 100
print("Modified tensor:", tensor_sample)

#Slicing a tensor
# Slice tensor_sample

subset_tensor_sample = tensor_sample[1:4]
print("Original tensor sample: ", tensor_sample)
print("The subset of tensor sample:", subset_tensor_sample)

print("Inital value on index 3 and index 4:", tensor_sample[3:5])
tensor_sample[3:5] = torch.tensor([300.0, 400.0])
print("Modified tensor:", tensor_sample)

# Using variable to contain the selected index, and pass it to slice operation

selected_indexes = [3, 4, 7]
subset_tensor_sample = tensor_sample[selected_indexes]
print("The inital tensor_sample", tensor_sample)
print("The subset of tensor_sample with the values on index 3 and 4: ", subset_tensor_sample)

#Statistics on tensors
math_tensor = torch.tensor([1.0, -1.0, 1, -1])
print("Tensor example: ", math_tensor)
mean = math_tensor.mean()
print("The mean of math_tensor: ", mean)
standard_deviation = math_tensor.std()
print("The standard deviation of math_tensor: ", standard_deviation)

max_min_tensor = torch.tensor([1, 1, 3, 5, 5])
print("Tensor example: ", max_min_tensor)
max_val = max_min_tensor.max()
print("Maximum number in the tensor: ", max_val)
min_val = max_min_tensor.min()
print("Minimum number in the tensor: ", min_val)

# Method for calculating the sin result of each element in the tensor

pi_tensor = torch.tensor([0, np.pi/2, np.pi])
sin = torch.sin(pi_tensor)
print("The sin result of pi_tensor: ", sin)

#A useful function for plotting mathematical functions is torch.linspace(). 
#torch.linspace() returns evenly spaced numbers over a specified interval. 
#You specify the starting point of the sequence and the ending point of the sequence. T
#he parameter steps indicates the number of samples to generate. Now, you'll work with steps = 5.

# Construct the tensor within 0 to 360 degree


len_5_tensor = torch.linspace(-2, 2, steps = 5)
print ("First Try on linspace", len_5_tensor)

pi_tensor = torch.linspace(0, 2*np.pi, 100)
sin_result = torch.sin(pi_tensor)

# Plot sin_result

plt.plot(pi_tensor.numpy(), sin_result.numpy())

#TENSOR OPERATIONS
