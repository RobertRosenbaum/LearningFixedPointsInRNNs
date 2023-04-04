import torch.nn as nn
import torch

# This is a recurrent, dynamical rate model, as used often
# in comp neuro. It comes from an ODE, but this is a discrete
# time version equivalent to solving the ODE using forward Euler
# with eta=dt/tau
class RateModel(nn.Module):

    def __init__(self, N, Nt, rho, Nx, eta=.05, nonlinearity='tanh'):
        super(RateModel, self).__init__()
        self.N = N
        self.Nt = Nt
        self.rho = rho
        self.Nx = Nx

        # activation == fI curve, f, can be a string for relu or tanh
        # or it can be any function
        if nonlinearity=='relu':
            self.f = torch.relu
        elif nonlinearity=='tanh':
            self.f = torch.tanh
        else:
            self.f = nonlinearity

        self.eta = eta
        self.WT = nn.Parameter(self.rho*torch.randn(N,N)/torch.sqrt(torch.tensor(N)))
        self.WxT = nn.Parameter(torch.randn(Nx,N)/torch.sqrt(torch.tensor(Nx)), requires_grad=False)

    def forward(self, x):
        batch_size = x.shape[0]
        r = torch.zeros(batch_size,self.N, requires_grad=True).to(self.WT.device)
        xWxT = x@self.WxT
        for i in range(self.Nt):
            r = r + self.eta * (-r + self.f(r@self.WT+xWxT))
        return r


# This is a different type of rate model, as used
# commonly in comp neuro
class RateModelType2(nn.Module):

    def __init__(self, N, Nt, rho, Nx, eta=.05, nonlinearity='tanh'):
        super(RateModelType2, self).__init__()
        self.N = N
        self.Nt = Nt
        self.rho = rho
        self.Nx = Nx

        # activation == fI curve, f, can be a string for relu or tanh
        # or it can be any function
        if nonlinearity=='relu':
            self.f = torch.relu
        elif nonlinearity=='tanh':
            self.f = torch.tanh
        else:
            self.f = nonlinearity

        self.eta = eta
        self.WT = nn.Parameter(self.rho*torch.randn(N,N)/torch.sqrt(torch.tensor(N)))
        self.WxT = nn.Parameter(torch.randn(Nx,N)/torch.sqrt(torch.tensor(Nx)), requires_grad=False)

    def forward(self, x):
        batch_size = x.shape[0]
        v = torch.zeros(batch_size,self.N, requires_grad=True).to(self.WT.device)
        xWxT = x@self.WxT
        for i in range(self.Nt):
            v = v + self.eta * (-v + self.f(v)@self.WT+xWxT)
        return self.f(v)




# This is a discrete-time RNN, as typically used in machine learning
# We could use torch.rnn instead of this explicit implementation
# but let's just use this for now.
class RNNModel(nn.Module):

    def __init__(self, N, Nt, rho, Nx, eta=.05, nonlinearity='tanh'):
        super(RNNModel, self).__init__()
        self.N = N
        self.Nt = Nt
        self.rho = rho
        self.Nx = Nx

        # activation == fI curve, f, can be a string for relu or tanh
        # or it can be any function
        if nonlinearity=='relu':
            self.f = torch.relu
        elif nonlinearity=='tanh':
            self.f = torch.tanh
        else:
            self.f = nonlinearity

        self.eta = eta
        self.WT = nn.Parameter(self.rho*torch.randn(N,N)/torch.sqrt(torch.tensor(N)))
        self.WxT = nn.Parameter(torch.randn(Nx,N)/torch.sqrt(torch.tensor(Nx)), requires_grad=False)

    def forward(self, x):
        batch_size = x.shape[0]
        r = torch.zeros(batch_size,self.N, requires_grad=True).to(self.WT.device)
        xWxT = x@self.WxT
        for i in range(self.Nt):
            r = self.f(r@self.WT+xWxT)
        return r
