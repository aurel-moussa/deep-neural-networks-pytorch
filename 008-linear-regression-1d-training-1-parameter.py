#We will only train one parameter here for this linear equation, namely the slope
#IMPORTING LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import torch

#HELPER FUNCTION FOR VISUALIZATION
# The class for plotting

class plot_diagram():
    
    # Constructor
    def __init__(self, X, Y, w, stop, go = False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values] 
        w.data = start
        
    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y,'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)   
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
    
    # Destructor
    def __del__(self):
        plt.close('all')
        
#Generate values from -3 to 3 that create a line with a slope of -3. This is the line we will estimate.
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X

# Plot the line with blue
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Add some noise to f(X) and save it in Y to make th e output data more realistic with that added noise
Y = f + 0.1 * torch.randn(X.size()) #this is from a standard normal distribution

# Plot the data points
plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'Y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#MODEL AND COST FUNCTION (TOTAL LOSS)
#et us create the model and the cost function (total loss) we are going to use to train the model and evaluate the result.
#Let us create our own forwad function
# Create forward function for prediction

def forward(x):
    return w * x

#define our cost or criterion function
# as a mean Ssquared Error function to evaluate the result.

def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

# Create Learning Rate and an empty list to record the loss for each iteration
lr = 0.1
LOSS = []

#Now, we create a model parameter by setting the argument requires_grad to  True because the system must learn it.
w = torch.tensor(-10.0, requires_grad = True)

#Create a plot_diagram object to visualize the data space and the parameter space for each iteration during training:
gradient_plot = plot_diagram(X, Y, w, stop = 5)

#TRAINING MODEL FUNCTION
# Define a function for train the model

def train_model(iter):
    for epoch in range (iter):
        
        # make the prediction as we learned in the last lab
        Yhat = forward(X)
        
        # calculate the iteration
        loss = criterion(Yhat,Y)
        
        # plot the diagram for us to have a better idea
        gradient_plot(Yhat, w, loss.item(), epoch)
        
        # store the loss into list
        LOSS.append(loss.item())
        
        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        
        # updata parameters
        w.data = w.data - lr * w.grad.data
        
        # zero the gradients before running the backward pass
        w.grad.data.zero_()
