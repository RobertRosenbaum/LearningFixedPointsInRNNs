import torch
import torch.nn.functional as functional

def Set_Grad_dW3(model, Loss, r, X, Y, Yhat, RT):
    N = r.shape[1]
    device = r.device
    dW = torch.zeros(N, N).to(device)
    Id = torch.eye(N).to(device)
    g = 1 - r ** 2
    s = functional.softmax(Yhat, dim=1)
    W = model.WT.T
    yoh = functional.one_hot(Y, num_classes=10)
    for ii in range(len(X)):
        # vector of gains
        gi = g[ii, :]

        # These don't seem right, but they are. See next code cell
        # where they are tested
        GiW = W * gi[:, None]
        ImWGiGi = gi[None, :] * (Id - gi[None, :] * W)
        ImGiW = Id - GiW

        dW += torch.outer(ImWGiGi @ (RT) @ (s[ii, :] - yoh[ii, :]),
                          (ImGiW.T) @ ImGiW @ r[ii, :]) / len(X)

    model.WT.grad = dW.T

def Set_Grad_dW3_relu(model, Loss, r, X, Y, Yhat, RT):
    N = r.shape[1]
    device = r.device
    dW = torch.zeros(N, N).to(device)
    Id = torch.eye(N).to(device)
    # g = 1 - r ** 2
    g = 1*(r>0)
    s = functional.softmax(Yhat, dim=1)
    W = model.WT.T
    yoh = functional.one_hot(Y, num_classes=10)
    for ii in range(len(X)):
        # vector of gains
        gi = g[ii, :]

        # These don't seem right, but they are. See next code cell
        # where they are tested
        GiW = W * gi[:, None]
        ImWGiGi = gi[None, :] * (Id - gi[None, :] * W)
        ImGiW = Id - GiW

        dW += torch.outer(ImWGiGi @ (RT) @ (s[ii, :] - yoh[ii, :]),
                          (ImGiW.T) @ ImGiW @ r[ii, :]) / len(X)

    model.WT.grad = dW.T

def Set_Grad_BPTT(model, Loss, r, X, Y, Yhat, RT):
    Loss.backward()

def Set_Grad_dW1(model, Loss, r, X, Y, Yhat, RT):
    N = r.shape[1]
    device = r.device
    dW = torch.zeros(N, N).to(device)
    Id = torch.eye(N).to(device)
    g = 1 - r ** 2
    s = functional.softmax(Yhat, dim=1)
    W = model.WT.T
    yoh = functional.one_hot(Y, num_classes=10)
    for ii in range(len(X)):
        # vector of gains
        gi = g[ii, :]
        # Gi = torch.diag(gi)

        GiInvImWTGi = torch.linalg.inv(Id - gi[None,:]*W.T)*gi[:,None]

        dW+=torch.outer((GiInvImWTGi)@(RT)@(s[ii,:]-yoh[ii,:]), r[ii,:])/len(X)

    model.WT.grad = dW.T

def Set_Grad_dW1_relu(model, Loss, r, X, Y, Yhat, RT):
    N = r.shape[1]
    device = r.device
    dW = torch.zeros(N, N).to(device)
    Id = torch.eye(N).to(device)
    # g = 1 - r ** 2
    g = 1*(r>0)
    s = functional.softmax(Yhat, dim=1)
    W = model.WT.T
    yoh = functional.one_hot(Y, num_classes=10)
    for ii in range(len(X)):
        # vector of gains
        gi = g[ii, :]
        # Gi = torch.diag(gi)

        GiInvImWTGi = torch.linalg.inv(Id - gi[None,:]*W.T)*gi[:,None]

        dW+=torch.outer((GiInvImWTGi)@(RT)@(s[ii,:]-yoh[ii,:]), r[ii,:])/len(X)

    model.WT.grad = dW.T