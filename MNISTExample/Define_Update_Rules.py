import torch
import torch.nn.functional as functional

def Set_Grad_dW3(model, Loss, r, X, Y, Yhat, RT, gain = 'tanh'):
    N = r.shape[1]
    device = r.device
    dW = torch.zeros(N, N).to(device)
    Id = torch.eye(N).to(device)

    if gain == 'tanh':
        g = 1.0 - r ** 2
    elif gain == 'relu':
        g = 1.0 * (r > 0)
    else:
        g = gain

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


def Set_Grad_BPTT(model, Loss, r, X, Y, Yhat, RT, gain = 'tanh'):
    Loss.backward()

def Set_Grad_dW1(model, Loss, r, X, Y, Yhat, RT, gain = 'tanh'):
    N = r.shape[1]
    device = r.device
    dW = torch.zeros(N, N).to(device)
    Id = torch.eye(N).to(device)

    if gain == 'tanh':
        g = 1.0 - r ** 2
    elif gain == 'relu':
        g = 1.0 * (r > 0)
    else:
        g = gain

    s = functional.softmax(Yhat, dim=1)
    W = model.WT.T
    yoh = functional.one_hot(Y, num_classes=10)
    for ii in range(len(X)):
        # vector of gains
        gi = g[ii, :]


        GiInvImWTGi = torch.linalg.inv(Id - gi[None,:]*W.T)*gi[:,None]

        # # Check for correctness
        # GiInv = torch.diag(1/gi)
        # GiInvImWTGiLong = torch.linalg.inv(GiInv - W).T
        # print('dW1 test:',(GiInvImWTGi-GiInvImWTGiLong).norm()/GiInvImWTGiLong.norm())

        dW+=torch.outer((GiInvImWTGi)@(RT)@(s[ii,:]-yoh[ii,:]), r[ii,:])/len(X)

    model.WT.grad = dW.T



def Set_Grad_dW4(model, Loss, r, X, Y, Yhat, RT, gain = 'tanh'):
    N = r.shape[1]
    device = r.device
    dW = torch.zeros(N, N).to(device)
    Id = torch.eye(N).to(device)

    if gain == 'tanh':
        g = 1.0 - r ** 2
    elif gain == 'relu':
        g = 1.0 * (r > 0)
    else:
        g = gain

    s = functional.softmax(Yhat, dim=1)
    W = model.WT.T
    yoh = functional.one_hot(Y, num_classes=10)
    for ii in range(len(X)):
        # vector of gains
        gi = g[ii, :]

        # These don't seem right, but they are. See next code cell
        # where they are tested
        ImWGi =  (Id - gi[None, :] * W)


        dW += torch.outer(ImWGi @ (RT) @ (s[ii, :] - yoh[ii, :]),
                          (ImWGi.T) @ ImWGi @ r[ii, :]) / len(X)

    model.WT.grad = dW.T

