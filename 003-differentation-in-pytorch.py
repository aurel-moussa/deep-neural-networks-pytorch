#Library imports

import torch 
import matplotlib.pylab as plt


#DERIVATES
# Create a tensor x, setting requires grad to true to state that we will be taking derivates for this tensor.
#The value of 2 is just set as we are interested in the results of the derivate if the value is 2

x = torch.tensor(2.0, requires_grad = True)
print("The tensor x: ", x)

# Create a tensor y according to y = x^2. Y is the function that we will want to to derivates from
y = x ** 3
print("The result of y = x^2: ", y)

# Take the derivative. Try to print out the derivative at the value x = 2

y.backward() #This creates the derivate i.e. dy(x)/dx = 2x. Since the derivate of x² is equal to 2x. 
#The derivate of something like x³+2x would be euqal to 3x + 2

print("The dervative at x = 2: ", x.grad) #The x-grad then takes the specific value of x. Here it is dy(x=2)/d(x) = 2(2) = 4

#What are the settings/parameters for the tensors?
print('data:',x.data)
print('grad_fn:',x.grad_fn)
print('grad:',x.grad)
print("is_leaf:",x.is_leaf)
print("requires_grad:",x.requires_grad)

print('data:',y.data)
print('grad_fn:',y.grad_fn)
print('grad:',y.grad)
print("is_leaf:",y.is_leaf)
print("requires_grad:",y.requires_grad)

#More complicated function to create derivate from
x = torch.tensor(2.0, requires_grad = True)
y = x ** 2 + 2 * x + 1
print("The result of y = x^2 + 2x + 1: ", y)
y.backward() #Creates derivate, i.e., 2x + 2
print("The dervative at x = 2: ", x.grad) #Result of this derviate function at the value of x of 2 i.e. 2(2) + 2 = 6

# We can implement our own custom autograd Functions by s
#ubclassing torch.autograd.Function and implementing the forward and backward passes which operate on Tensors

class SQ(torch.autograd.Function):


    @staticmethod
    def forward(ctx,i):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        result=i**2
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        i, = ctx.saved_tensors
        grad_output = 2*i
        return grad_output
      
      
x=torch.tensor(2.0,requires_grad=True )
sq=SQ.apply

y=sq(x)
y
print(y.grad_fn)
y.backward()
x.grad      

#PARTIAL DERIVATES
#artial Derivatives. Consider the function: 𝑓(𝑢,𝑣)=𝑣𝑢+𝑢2

# Calculate f(u, v) = v * u + u^2 at u = 1, v = 2

u = torch.tensor(1.0,requires_grad=True)
v = torch.tensor(2.0,requires_grad=True)
f = u * v + u ** 2
print("The result of v * u + u^2: ", f)

# Calculate the derivative with respect to u
f.backward()
print("The partial derivative with respect to u: ", u.grad)
#∂f(u,v)∂𝑢=𝑣+2𝑢
#∂f(u=1,v=2)∂𝑢=2+2(1)=4

# Calculate the derivative with respect to v
print("The partial derivative with respect to u: ", v.grad)
#∂f(u,v)∂𝑣=𝑢
#∂f(u=1,v=2)∂𝑣=1

#CALCULATING DERIVATES FOR FUNCTIONS WIHT MULTIPLE VALUES:
#Calculate the derivative with respect to a function with multiple values as follows. 
#You use the sum trick to produce a scalar valued function and then take the gradient

# Calculate the derivative with multiple values

x = torch.linspace(-10, 10, 10, requires_grad = True)
Y = x ** 2
y = torch.sum(x ** 2)

# Take the derivative with respect to multiple value. Plot out the function and its derivative

y.backward()

plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

## Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative

x = torch.linspace(-10, 10, 1000, requires_grad = True)
Y = torch.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

y.grad_fn
