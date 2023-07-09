import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle

from Model_Definitions import RateModel as ModelType
from Define_Update_Rules import Set_Grad_dW1 as Set_Grad
from Train_MNIST_Model import Train_MNIST_Model
from Create_Plots import CreatePlots

# Set hyper-params
from Shared_Hyperparams import *

with torch.no_grad():

    # Set to True if you want to rerun sims,
    # False if you just want to load data for plots
    RunSims = True

    LearningRates = [.001, .0025, .005, .01, .025]

    # Regularization coefficient
    alpha=0

    print(N,Nt,rho,Nx,eta,nonlinearity,train_batch_size,num_epochs,alpha)



    # Define a function that returns an instance of the model
    def Get_Model():
        return ModelType(N, Nt, rho, Nx, eta, nonlinearity)

    # Define readout matrix
    readout_matrix = torch.randn(10,N)/np.sqrt(N)

    ### IF runsims then save Training_Stats, eigenvalues

    # V: save objects using pickle
    def save_objects(obj1, obj2, filename):
        objects = (obj1, obj2)
        with open(filename, 'wb') as f:
            pickle.dump(objects, f)

    # V: read objects using pickle
    def read_objects(filename):
        with open(filename, 'rb') as f:
            objects = pickle.load(f)
        obj1, obj2 = objects

        return obj1, obj2

    fname='./data/MNISTFixedPoints_dW1Ratetanh.npy'

    if RunSims:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
      # Train model
            model,Training_Stats=Train_MNIST_Model(LearningRates, Get_Model, readout_matrix,
                                                   train_batch_size, test_batch_size,
                                                   num_epochs, Set_Grad, nonlinearity, alpha)

    ### save to filename matching this file name
            save_objects(model, Training_Stats, fname)
    else:
        model, Training_Stats = read_objects(fname)

    # Make plots
    CreatePlots(Training_Stats)
    plt.show()
    print('done')
