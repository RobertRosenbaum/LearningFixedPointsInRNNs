import torch
import numpy as np
import matplotlib.pyplot as plt

def CreatePlots(Training_Stats):

    print('Making plots.')

    Losses = Training_Stats['TrainingLosses']
    Accuracies = Training_Stats['TrainingAccuracies']
    TestLosses = Training_Stats['TestLosses']
    TestAccuracies = Training_Stats['TestAccuracies']

    plt.figure()
    plt.plot(Losses.T)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.ylim([0, 2.5])

    plt.figure()
    plt.plot(TestLosses.T)
    plt.xlabel("Iteration")
    plt.ylabel("Test Loss")
    plt.ylim([0, 2.5])

    plt.figure()
    plt.plot(100 * Accuracies.T)
    plt.xlabel("Iteration")
    plt.ylabel("Training Accuracy (%)")
    plt.ylim([0, 100])

    plt.figure()
    plt.plot(100 * TestAccuracies.T)
    plt.xlabel("Iteration")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim([0, 100])

    plt.show()
