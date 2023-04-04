import torch
import torch.nn as nn
import numpy as np
from time import time as tm
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def Train_MNIST_Model(LearningRates, Get_Model, readout_matrix, train_batch_size, test_batch_size, num_epochs, Set_Grad, alpha = 0, seed=1):

    print('Training model.')


    # Load training and testing data from MNIST dataset
    train_dataset = MNIST('./',
          train=True,
          transform=transforms.ToTensor(),
          download=True)

    test_dataset = MNIST('./',
          train=False,
          transform=transforms.ToTensor(),
          download=True)

    # Print the size of the two data sets
    m = len(train_dataset)
    mtest = len(test_dataset)

    # train_dataset.data contains all the MNIST images (X)
    # train_dataset.targets contains all the labels (Y)
    print("Size of training inputs (X)=",train_dataset.data.size())
    print("Size of training labels (Y)=",train_dataset.targets.size())



    # Data loader. These make it easy to iterate through batches of data.
    # Shuffle=True means that the data will be randomly shuffled on every epoch
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=True)


    # Use cross-entropy loss.
    # This means we will use the softmax loss function
    MySoftMaxLoss = nn.CrossEntropyLoss()

    # Transpose of readout matrix
    RT = readout_matrix.T

    # Define the device to use.
    # Use 'cpu' for cpu and 'cuda' for gpu
    # We will discuss GPUs later, so leave this as cpu for now
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device=', device)

    # Compute number of steps per epoch
    # and total number of steps
    steps_per_epoch = len(train_loader)
    total_num_steps = num_epochs * steps_per_epoch
    print("steps per epoch=", steps_per_epoch, "\nnum epochs=", num_epochs, "\ntotal number of steps=", total_num_steps)

    # Initialize Test Losses if we'll compute them
    TestLosses = np.zeros((len(LearningRates), total_num_steps))
    TestAccuracies = np.zeros((len(LearningRates), total_num_steps))

    # Initialize training losses to plot
    Losses = np.zeros((len(LearningRates), total_num_steps))
    Accuracies = np.zeros((len(LearningRates), total_num_steps))

    t1 = tm()  # Start the timer
    for kk in range(len(LearningRates)):
        torch.manual_seed(seed)
        model = Get_Model().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LearningRates[kk], weight_decay=alpha)
        j = 0  # Counter to keep track of iterations
        for k in range(num_epochs):
            # Re-initializes the training iterator (shuffles data for one epoch)
            TrainingIterator = iter(train_loader)

            for i in range(steps_per_epoch):

                X, Y = next(TrainingIterator)
                X = (X.reshape(-1, 28 * 28)).to(device)
                Y = Y.to(device)

                # Forward pass: compute yhat and loss for this batch
                r = model(X)
                N = model.N
                Yhat = r @ RT

                Loss = MySoftMaxLoss(Yhat, Y)

                # Compute accuracy
                with torch.no_grad():
                    Losses[kk, j] = Loss.item()
                    PredictedClass = torch.argmax(Yhat, dim=1)
                    Accuracies[kk, j] = (PredictedClass == Y).float().mean().item()

                    # Compute test losses if flag is on
                    TestingIterator = iter(test_loader)
                    X, Y = next(TestingIterator)
                    X = (X.reshape(-1, 28 * 28)).to(device)
                    X = X.to(device)
                    Y = Y.to(device)
                    r = model(X)
                    Yhat = r @ RT
                    TestLosses[kk, j] = MySoftMaxLoss(Yhat, Y).item()
                    PredictedClass = torch.argmax(Yhat, dim=1)
                    TestAccuracies[kk, j] = (PredictedClass == Y).float().mean().item()

                with torch.no_grad():
                    # # Backward pass and optimize
                    # optimizer.zero_grad() # Zero-out gradients from last iteration
                    # Loss.backward()       # Compute gradients

                    # optimizer.step()      # Update parameters

                    Set_Grad(model, Loss, r, X, Y, Yhat, RT)

                    # dW = torch.zeros(N, N).to(device)
                    # Id = torch.eye(N).to(device)
                    # g = 1 - r ** 2
                    # s = functional.softmax(Yhat, dim=1)
                    # W = model.WT.T
                    # yoh = functional.one_hot(Y, num_classes=10)
                    # for ii in range(len(X)):
                    #     # vector of gains
                    #     gi = g[ii, :]
                    #
                    #     # These don't seem right, but they are. See next code cell
                    #     # where they are tested
                    #     GiW = W * gi[:, None]
                    #     ImWGiGi = gi[None, :] * (Id - gi[None, :] * W)
                    #     ImGiW = Id - GiW
                    #
                    #     dW += torch.outer(ImWGiGi @ (RT) @ (s[ii, :] - yoh[ii, :]),
                    #                       (ImGiW.T) @ ImGiW @ r[ii, :]) / len(X)
                    # dW += alpha * W
                    # model.WT.grad = dW.T

                    #model.WT -= LearningRates[kk] * dW.T
                    optimizer.step()


                j += 1
            print('LR [{}/{}], Epoch [{}/{}], Loss: {:.2f}, Acc: {:.2f}%, time:{:.1f}'.format(kk + 1, len(LearningRates), k + 1,
                                                                                 num_epochs, Loss.item(),
                                                                                 100 * Accuracies[kk, j - 1],tm()-t1),flush=True)

    TrainingTime = tm() - t1
    print("Training time =", TrainingTime, "sec =", TrainingTime / (len(LearningRates) * num_epochs), "sec/epoch =",
          TrainingTime / (len(LearningRates) * total_num_steps), "sec/step")

    # Compute and print number of trainable parameters
    NumParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters in model =', NumParams)

    print('Final Training Losses:', Losses[:, j - 1], '%')
    print('Final Training Accuracy:', 100 * Accuracies[:, j - 1], '%')

    print('Final Test Loss:', TestLosses[:, j - 1])
    print('Final Test Accuracy:', 100 * TestAccuracies[:, j - 1], '%')

    torch.cuda.empty_cache()  # This helps avoid out-of-memory errors

    Training_Stats = {'TrainingLosses':Losses, 'TrainingAccuracies':Accuracies, 'TestLosses':TestLosses, 'TestAccuracies':TestAccuracies, 'TrainingTime':TrainingTime}


    return model,Training_Stats
