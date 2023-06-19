
print('Setting shared hyperparameters.')

# Number of hidden units
N=300

# Number of time steps
Nt=500

# Step size for rate models
# (equiv to dt/tau)
eta=.01


# Number of epochs and batch sizes
num_epochs = 3        # Number of times to go through training data
train_batch_size = 512      # Batch size to use with training data
test_batch_size = 512  # Batch size to use for test data

# Spectral radius of initial hidden connectivity matrix
rho=0.5

# Activation function/f-I curve to use
nonlinearity = 'tanh'

# Input size
Nx = 28*28


