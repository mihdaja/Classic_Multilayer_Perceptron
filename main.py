import torch 
import torch.nn as nn 

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs): # num inputs reprezents the number of inputs, for example 28x28 in the case of the mnist dataset i'll use
        super().__init__() # this allows me to use the methods in nn.Module, which are very useful for making nn's

        # Using Sequential is not required, but it can make our life easier if we have a series of layers that we want to execute in a specific order, as is the case here. This way, after instantiating self.layers = Sequential(...) in the __init__ constructor, we just have to call the self.layers instead of calling each layer individually in the NeuralNetwork’s forward method.

        self.layers = nn.Sequential( # initializes a sequential container, which will be important for using the .forward method later on to move the input data through the network

            # first hidden layer
            nn.Linear(num_inputs, 30), # takes in the number of inputs orderd in a matrix: A(1,num_inputs) then passes out a matrix: A1(1, 30)

            nn.ReLU(), # passes the matrix in only positives (0 if x < 0 and x if x >= 0), easier to differentiate than a sigmoid

            #second hidden layer
            nn.Linear(30, 20),
            nn.ReLU(),
            
            # output layer
            nn.Linear(20, num_outputs), 
        )

    def forward(self, x): # passes the input through all the neural layers and outputs the resulting logits through the return function
        logits = self.layers(x)
        return logits
        

model = NeuralNetwork(28*28, 10)

# print(model)

# model(x) outputs the logits from the forward function

param_nr = sum(
    p.numel() for p in model.parameters() if p.requires_grad
) # p.numel() returns the total number of trainable parameters in a tensor, and p.requires_grad checks if the parameter is trainable

# print (f"Nr of trainable params: {param_nr}")

# based on the print(model) output, the first Linear layers is at pos 0 in the layers attribute

#print(model.layers[0].weight) # see the parameters in the first Linear layer

#print(model.layers[0].weight.shape) # see the shape of the first Linear layer

X = torch.rand((1,28*28)) #generates a random input matrix for the model to check it works

out = model(X) # model(X) automatically calls forward(X)
#print(out) # the grad_fn=<AddmmBackward0> calculates the last made calculation in order to use this information when it computes the differentiation for backpropagation

#after training however, this is useless, so we would use something like this in order to save on processing power
#with torch.no_grad():
#   out = model(X)
#print(out)

#also, we don't use a softmax function on the loss, because the commonly used torch loss functions apply this function in the call, making it unnecessary in the training. If we want to use the model with softmax after training in order to produce predictions we'd use something like this:
#with torch.no_grad():
#   out = torch.softmax(model(X), dim = 1)
#print(out)