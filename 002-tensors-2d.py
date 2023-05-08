#Basics of tensor operations of 2D tensors
#Library import

import numpy as np 
import matplotlib.pyplot as plt
import torch
import pandas as pd

#Converting a 2D Python list into a 2D tensor
twoD_list = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
twoD_tensor = torch.tensor(twoD_list)
print("The New 2D Tensor: ", twoD_tensor)

print("The dimension of twoD_tensor: ", twoD_tensor.ndimension())
print("The shape of twoD_tensor: ", twoD_tensor.shape)
print("The shape of twoD_tensor: ", twoD_tensor.size())
print("The number of elements in twoD_tensor: ", twoD_tensor.numel())

# Convert tensor to numpy array; Convert numpy array to tensor

twoD_numpy = twoD_tensor.numpy()
print("Tensor -> Numpy Array:")
print("The numpy array after converting: ", twoD_numpy)
print("Type after converting: ", twoD_numpy.dtype)

print("================================================")

new_twoD_tensor = torch.from_numpy(twoD_numpy)
print("Numpy Array -> Tensor:")
print("The tensor after converting:", new_twoD_tensor)
print("Type after converting: ", new_twoD_tensor.dtype)

#convert Panda Dataframe to tensor

df = pd.DataFrame({'a':[11,21,31],'b':[12,22,312]})

print("Pandas Dataframe to numpy: ", df.values)
print("Type BEFORE converting: ", df.values.dtype)

print("================================================")

new_tensor = torch.from_numpy(df.values)
print("Tensor AFTER converting: ", new_tensor)
print("Type AFTER converting: ", new_tensor.dtype)

#INDEXING
#Indexing is done like with lists, e.g., Array[0][3] refers to taking the first row, and then the fourth column in that row
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 2nd-row 3rd-column? ", tensor_example[1, 2]) #one way of indexing
print("What is the value on 2nd-row 3rd-column? ", tensor_example[1][2]) #the other way of indexing

#Slicing
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 1st-row first two columns? ", tensor_example[0, 0:2])
print("What is the value on 1st-row first two columns? ", tensor_example[0][0:2])

#But we can't combine using slicing on row and pick one column 
#by using the code tensor_obj[begin_row_number: end_row_number][begin_column_number: end_column number]. 
#The reason is that the slicing will be applied on the tensor first. 
#The result type will be a two dimension again. 
#The second bracket will no longer represent the index of the column it will be the index of the row at that time. Let us see an example.

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
sliced_tensor_example = tensor_example[1:3]
print("1. Slicing step on tensor_example: ")
print("Result after tensor_example[1:3]: ", sliced_tensor_example)
print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension())
print("================================================")
print("2. Pick an index on sliced_tensor_example: ")
print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1])
print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension())
print("================================================")
print("3. Combine these step together:")
print("Result: ", tensor_example[1:3][1])
print("Dimension: ", tensor_example[1:3][1].ndimension())

#However, this indexing works:
# Use tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number] , ie using the comma

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 3rd-column last two rows? ", tensor_example[1:3, 2])

#TENSOR OPERATIONS
#Tensor addition for 2D takes the values of the same column-row and adds them together
# Calculate [[1, 0], [0, 1]] + [[2, 1], [1, 2]]

X = torch.tensor([[1, 0],[0, 1]]) 
Y = torch.tensor([[2, 1],[1, 2]])
X_plus_Y = X + Y
print("The result of X + Y: ", X_plus_Y)

#Scalar times matrix
# Calculate 2 * [[2, 1], [1, 2]]
Y = torch.tensor([[2, 1], [1, 2]]) 
two_Y = 2 * Y
print("The result of 2Y: ", two_Y)

#elementwise product / hadamard product
X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]]) 
X_times_Y = X * Y
print("The result of X * Y: ", X_times_Y)

#The Matrix Multiplication from Linear Algebra
#in the multiplication of two matrices order matters. 
#This means if X * Y is valid, it does not mean Y * X is valid. 
#The number of columns of the matrix on the left side of the multiplication sign 
#must equal to the number of rows of the matrix on the right side.
#We use torch.mm() for calculating the multiplication between tensors with different sizes.

# Calculate [[0, 1, 1], [1, 0, 1]] * [[1, 1], [1, 1], [-1, 1]]

A = torch.tensor([[0, 1, 1], 
                  [1, 0, 1]])
B = torch.tensor([[1, 1], 
                  [1, 1], 
                  [-1, 1]])
A_times_B = torch.mm(A,B)
#First row, first column * First row, first column (0*1)
#First row, second column * Second row, first column (1*1)
#First row, third column * Third Row, first column (1*-1) 
#Equals first row, first column = 0

#First row, first column * First row, second column (0*1)
#First row, second column * Seocnd row, second column (1*1)
#First row, third column * Third row, second column (1*1)
#Equals first row, second column = 2


print("The result of A * B: ", A_times_B)
