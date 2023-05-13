#IMPORTING LIBRARIES
import torch

#Let's create these equations
#𝑏ias=−1,𝑤eight=2
#𝑦̂ =−1+2𝑥

# Define w = 2 and b = -1 for y = wx + b
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)

#Our manual prediction according to the equation for yhat
# Function forward(x) for prediction

def forward(x):
    yhat = w * x + b
    return yhat
  
#What is our prediction of yhat if x = 1?  
# Predict y = 2x - 1 at x = 1

x = torch.tensor([[1.0]])
yhat = forward(x)
print("The prediction: ", yhat)

#We can also make predictions for multiple values of x
# Create x Tensor and check the shape of x tensor

x = torch.tensor([[1.0], [2.0]])
print("The shape of x: ", x.shape)

yhat = forward(x)
print("The prediction: ", yhat)

#Three predictions, wow, this is crazy
x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = forward(x)

#linear class can be used to make a prediction. 
#We can also use the linear class to build more complex models. Let's import the module:

# Import Class Linear
from torch.nn import Linear

# Set random seed manually, for testing purposes
torch.manual_seed(1)

# Create Linear Regression Model, and print out the parameters

lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

#We can also go through the parameters with the method state_dict()
print("Python dictionary: ",lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())

#keys correspond to the name of the attributes and the values correspond to the parameter value.
print("weight:",lr.weight)
print("bias:",lr.bias)

#Making a single prediction
# Make the prediction at x = [[1.0]]
x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

# Practice: Use the linear regression model object lr to make the prediction.
x = torch.tensor([[1.0],[2.0],[3.0]])

#CUSTOMIZED LINEAR REGRESSION CLASS
# Library for this section
from torch import nn
# Customize Linear Regression Class

class LR(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out
    
# Create the linear regression model. Print out the parameters.
lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)    

# Try our customize linear regression model with single input
x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

# Try our customize linear regression model with multiple input
x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)

print("Python dictionary: ", lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())
