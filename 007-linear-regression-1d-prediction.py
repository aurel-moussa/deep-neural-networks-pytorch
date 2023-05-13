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
