import torch
import numpy as np
from matplotlib import pyplot as plt
from fastonn import OpNetwork,utils,OpTier,OpBlock,SelfONN, Trainer
from fastonn.osl import *

#define the operation set library:
OPLIB = getOPLIB(
    NODAL = [mul, cubic, sine, expp, sinh, chirp],
    POOL = [sum],
    ACTIVATION =[tanh] 
)


print('type', type(OPLIB), str(OPLIB))

#configure ONN network:
model = OpNetwork(
    in_channels=1,
    tier_sizes=[12,], #number of neurons in hidden and input layer
    kernel_sizes=[21,],
    operators=[
        [1], #operators for 1st hidden layer
        
    ],
    sampling_factors= [2, ], #scaling factors for each layer
    OPLIB = OPLIB, #assign operator library
)


print(model)

#import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.optim as optim

loss = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(type(optimizer))



trainer = Trainer(model, trainloader, testloader, testloader,loss, utils.AdamFast, 0.001, {} ,"cpu",)
"""Initialize the trainer instance
        - **model** -- progress bar object to update  
        - **train_dl** -- training dataloader     
        - **val_dl** -- validation dataloader  
        - **test_dl** -- test dataloader  
        - **loss** -- loss function, must return a PyTorch tensor, not scalar  a
        - **opt_name** -- name of the optimizer. Either of  ['adamfast','cgd','adam','vanilla_adam']  
        - **lr** -- initial learning rate    
        - **metrics** -- Python dictionary with format 'metric_name':(func,max/min), where func is any function with inputs target,output and should return a scalar value for accuracy. max/min defines the desired optimization of this metric  
        - **device** -- device on which to train, either of ['cpu','cuda:x'] where x is the index of gpu. Multi-GPU training is not supported yet.    
        - **reset_fn** -- function to reset network parameters. Called at the start of each run  
        - **track** -- metric to track in format ['mode','metric_name','max/min']  
        - **model_name** -- filename of the saved model
        - **verbose** -- extent of debug output: 0=No output, 1=show only run progress bar, 2=show run and epoch progress bar  
        """
trainer.train()