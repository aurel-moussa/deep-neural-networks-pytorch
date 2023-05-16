
#LIBRARIES

import torch
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d

#Helper function to visualize data space
# The class for plot the diagram

class plot_error_surfaces(object):
    
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection = '3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    
    # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    
    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label = "training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
        
 # Set random seed for testing purposes
torch.manual_seed(1)

# Setup the actual data and simulated data
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

#have a visual look at this stuff
plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#MODEL AND COST FUNCTIONS
# Define the forward function
def forward(x):
    return w * x + b
  
# Define the MSE Loss function
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)  
  
# Create plot_error_surfaces for viewing the data
get_surface = plot_error_surfaces(15, 13, X, Y, 30)  

#TRAINING NORMAL MODEL
#Let us train the model
# Define the parameters w, b for y = wx + b, our beautiful little linear equation
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)

# Define learning rate and create an empty list for containing the loss for each iteration.
lr = 0.1
LOSS_BGD = []

# The function for training the model for NORMAL GRADIENT DESCENT (NOT STOACHISTIC!)

def train_model(iter):
    
    # Loop
    for epoch in range(iter):
        
        # make a prediction
        Yhat = forward(X)
        
        # calculate the loss 
        loss = criterion(Yhat, Y)

        # Section for plotting
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        get_surface.plot_ps()
            
        # store the loss in the list LOSS_BGD
        LOSS_BGD.append(loss)
        
        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        
        # update parameters slope and bias
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        
        # zero the gradients before running the backward pass
        w.grad.data.zero_()
        b.grad.data.zero_()
        
# Train the normal model with 10 iterations
train_model(10)  

#TRAINING STOCHAISTIC GRAD DESCENT MODEL
# Create plot_error_surfaces for viewing the data
get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

# The function for training the SGD model
LOSS_SGD = []
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)

def train_model_SGD(iter):
    
    # Loop
    for epoch in range(iter):
        
        # SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward(X)

        # store the loss 
        LOSS_SGD.append(criterion(Yhat, Y).tolist())
        
        for x, y in zip(X, Y): #here is where the stochaistic part starts, we are going through each individual sample one-by-one
            
            # make a pridiction
            yhat = forward(x)
        
            # calculate the loss 
            loss = criterion(yhat, y)

            # Section for plotting
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        
            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
        
            # update parameters slope and bias
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data

            # zero the gradients before running the backward pass
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        #plot surface and data space after each epoch    
        get_surface.plot_ps()
        
#Let us compare how well the SGD did, when compared to the normal full batch gradient descent
# Plot out the LOSS_BGD and LOSS_SGD

plt.plot(LOSS_BGD,label = "Batch Gradient Descent")
plt.plot(LOSS_SGD,label = "Stochastic Gradient Descent")
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()

#STOCHASTIC GRADIENT DESCENT USING DATA LOADER INSTEAD OF ZIP FUNCTION
# Import the library for DataLoader

from torch.utils.data import Dataset, DataLoader

# Create Dataset Class
class Data(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * self.x - 1
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):    
        return self.x[index], self.y[index]
    
    # Return the length
    def __len__(self):
        return self.len
      
# Create the dataset and check the length
my_beautiful_dataset = Data()
print("The length of dataset: ", len(my_beautiful_dataset)) 
print(my_beautiful_dataset[6]) #the sixth row (sample) of the dataset we just created

# Print the first sample
x, y = my_beautiful_dataset[0]
print("(", x, ", ", y, ")")

# Print the first 3 samples
x, y = my_beautiful_dataset[0:3]
print("The first 3 x: ", x)
print("The first 3 y: ", y)

# Create plot_error_surfaces for viewing the data
get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

# Create DataLoader
trainloader = DataLoader(dataset = my_beautiful_dataset, batch_size = 1)

#MODEL using the DATALOADER
# The function for training the model

w = torch.tensor(-15.0,requires_grad=True)
b = torch.tensor(-10.0,requires_grad=True)
LOSS_Loader = []

def train_model_DataLoader(epochs):
    
    # Loop
    for epoch in range(epochs):
        
        # SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward(X)
        
        # store the loss 
        LOSS_Loader.append(criterion(Yhat, Y).tolist())
        
        for x, y in trainloader: #instead of zip function
            
            # make a prediction
            yhat = forward(x)
            
            # calculate the loss
            loss = criterion(yhat, y)
            
            # Section for plotting
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            
            # Backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            # Updata parameters slope
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr* b.grad.data
            
            # Clear gradients 
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        #plot surface and data space after each epoch    
        get_surface.plot_ps()


# Run 10 iterations

train_model_DataLoader(10)        

#Plot the differences in errors after those iterations using the two methodos
plt.plot(LOSS_BGD,label="Batch Gradient Descent")
plt.plot(LOSS_Loader,label="Stochastic Gradient Descent with DataLoader")
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()

#Using a PyTorch DataLoader along with a dataset has several advantages over using the zip function directly. 
#Here are some key advantages:
"" Efficient Data Loading: The DataLoader class provides built-in multi-threading and memory management capabilities, allowing you to load and preprocess data in parallel while the model is training or predicting. It can automatically create multiple worker processes to load data in the background, optimizing data loading and minimizing the I/O overhead.

    Batching and Shuffling: The DataLoader class allows you to easily batch your data by specifying the batch size. It automatically collates samples into batches, which is especially useful when working with mini-batch stochastic gradient descent. Additionally, it provides options to shuffle the data during training, ensuring that the model sees different samples in each epoch and helps in reducing bias.

    Data Transformation and Augmentation: With a DataLoader, you can easily apply data transformations and augmentations using PyTorch's torchvision.transforms module. Transforms such as normalization, resizing, cropping, and flipping can be applied directly to the dataset during loading. This simplifies the data preprocessing pipeline and ensures consistency across training, validation, and testing.

    Integration with PyTorch Ecosystem: DataLoader seamlessly integrates with other PyTorch components, such as loss functions, optimizers, and GPU acceleration. It provides data in a format that is compatible with PyTorch models, making it easy to train, validate, and test your models using the PyTorch ecosystem.

    Flexibility and Customization: The DataLoader class offers a wide range of options and parameters that allow you to customize the data loading process according to your specific requirements. You can control the number of worker processes, enable/disable shuffling, set the drop last batch option, and more. Additionally, you can create your own custom collate function to handle complex data structures or specific batch processing needs.

While the zip function can be useful for basic pairing of multiple iterables, it lacks the advanced features and flexibility provided by the DataLoader class. When working with larger datasets or complex data loading scenarios, using DataLoader is highly recommended as it simplifies and optimizes the process of loading and preprocessing data for training deep learning models.""
