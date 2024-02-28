import torch
import numpy as np
from matplotlib import pyplot as plt
from fastonn import OpNetwork,utils,OpTier,OpBlock,Trainer,SelfONN
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
    tier_sizes=[12,32,1], #number of neurons in hidden and input layer
    kernel_sizes=[21,7,3],
    operators=[
        [1], #operators for 1st hidden layer
        [6], #operators for 2nd hidden layer
        [3] #output layer
    ],
    sampling_factors= [2, -2, 1], #scaling factors for each layer
    OPLIB = OPLIB, #assign operator library
)

print(model)