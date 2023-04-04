import torch.nn as nn
import torch

# Make a neural net. First define the functions
# that make up all the layers. Then compose the
# neural network as the "forward" pass
# The forward function is what gets called when you
# pass an input, x, to the network
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
