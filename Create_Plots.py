import torch
import numpy as np
import matplotlib.pyplot as plt

def CreatePlots(Training_Stats):
    print('Making plots.')
    LearningRates = Training_Stats['LearningRates']
    Losses = Training_Stats['TrainingLosses']
    Accuracies = Training_Stats['TrainingAccuracies']
    TestLosses = Training_Stats['TestLosses']
    TestAccuracies = Training_Stats['TestAccuracies']
    Jacobian = Training_Stats['Jacobian']

    # plt.figure()
    # plt.plot(Losses.T)
    # plt.xlabel("Iteration")
    # plt.ylabel("Training Loss")
    # plt.ylim([0, 2.5])

    # Plot training losses
    plt.figure()
    for kk in range(len(LearningRates)):
        plt.plot(Losses[kk].T, label=r'$\eta = $' + str(LearningRates[kk]))
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.ylim([0, 3])
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(TestLosses.T)
    # plt.xlabel("Iteration")
    # plt.ylabel("Test Loss")
    # plt.ylim([0, 2.5])

    # Plot testing losses
    plt.figure()
    for kk in range(len(LearningRates)):
        plt.plot(TestLosses[kk].T, label=r'$\eta = $' + str(LearningRates[kk]))
    plt.xlabel("Iteration")
    plt.ylabel("Test Loss")
    plt.ylim([0, 3])
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(100 * Accuracies.T)
    # plt.xlabel("Iteration")
    # plt.ylabel("Training Accuracy (%)")
    # plt.ylim([0, 100])

    # Plot training accuracy
    plt.figure()
    for kk in range(len(LearningRates)):
        plt.plot(100 * Accuracies[kk].T, label=r'$\eta = $' + str(LearningRates[kk]))
    plt.xlabel("Iteration")
    plt.ylabel("Training Accuracy (%)")
    plt.ylim([0, 100])
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(100 * TestAccuracies.T)
    # plt.xlabel("Iteration")
    # plt.ylabel("Test Accuracy (%)")
    # plt.ylim([0, 100])

    # Plot testing accuracy
    plt.figure()
    for kk in range(len(LearningRates)):
        plt.plot(100 * TestAccuracies[kk].T, label=r'$\eta = $' + str(LearningRates[kk]))
    plt.xlabel("Iteration")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim([0, 100])
    plt.legend()
    plt.show()

    # Plot Jacobian
    plt.figure()
    for kk in range(len(LearningRates)):
      plt.plot(np.real(Jacobian[kk]),np.imag(Jacobian[kk]),'.', label=r'$\eta = $' + str(LearningRates[kk]))
      plt.title('Eigenvalues of W after learning')
      plt.legend()
      plt.show()