# def load_data():
#     pass
# def gradient_descent_numpy(X, y, lr=0.01, steps=200):
#     pass

# def main():
#     pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from train import training

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data(test_split = 0.2):
    
    data = fetch_california_housing()
    x_train, x_test, \
    y_train, y_test = train_test_split(
                                        data.data, 
                                        data.target,
                                        test_size=test_split,
                                        random_state=0
                                    )

    scalar = StandardScaler()

    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)

    x_train = torch.tensor(x_train, dtype = torch.float32)
    x_test = torch.tensor(x_test, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype = torch.float32).reshape(-1, 1)


    '''
        IMPORTANT READ THIS:
        
        If you leave y_train as shape (n,) while your model outputs shape (n, 1), 
        MSELoss will silently broadcast them into an (n, n) matrix instead of erroring. 
        The loss still computes, still decreases, and is completely meaningless. 
        This is the easiest bug in the whole lab to miss, 
        because nothing crashes — check your shapes explicitly.
    ''';

    return x_train, y_train, x_test, y_test



def build_mlp(layers, act = 'relu'):
    in_dim, hidden, out_dim = layers[0], layers[1:-1], layers[-1]

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim, activation = act):
            super().__init__()
            self.fc_first = nn.Linear(in_dim, hidden[0])
            self.hidden = nn.ModuleList(
                                        [nn.Linear(hidden[i], hidden[i+1]) for i in range(len(hidden) - 1)]
                                    )
            self.fc_last = nn.Linear(hidden[-1], out_dim)

            self.relu = nn.ReLU()
            self.gelu = nn.GELU()
            self.swish = nn.SiLU()
            self.tanh = nn.Tanh()
            self.sign = torch.sign

            self.activation = self.activation_selector(activation.lower())


        def activation_selector(self, activation):
            if activation == 'relu':
                return self.relu
            elif activation == 'gelu':
                return self.gelu
            elif activation == 'swish':
                return self.swish
            elif activation == 'tanh':
                return self.tanh
            else:
                return self.sign
            

        def forward(self, x):
            x = self.fc_first(x)
            x = self.activation(x)
            for fc_hidden in self.hidden:
                x = fc_hidden(x)
                x = self.activation(x)
            x = self.fc_last(x)
            return x

    return MLP(in_dim, hidden, out_dim)


def main(x_train, y_train, x_test, y_test, epoch = 10):

    EPOCH = epoch
    BATCH_SIZE = len(x_train) # as for this we are just making 1 batch that contains the complete data

    train_data = TensorDataset(x_train, y_train)

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size

    train_data, val_data = random_split(
        train_data,
        [train_size, val_size]
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = build_mlp([8,32,1], act='relu')
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    model, history = training(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            epochs = EPOCH,
            optimizer = optimizer,
            loss_fn = loss_fn,
            device=device,
        )
    model.eval()

    with torch.no_grad():
        print('the test loss is - ', loss_fn(model(x_test.to(device)), y_test.to(device)).item())

    return model, history

    


def gradient_descent_numpy(X_train, y_train, lr=0.01, steps=200):
    
    w = np.zeros((X_train.shape[1], 1))
    b = 0.0

    lr = 0.01
    epochs = steps

    train_losses = []

    for epoch in range(epochs):

        # Forward pass
        y_pred = X_train @ w + b

        # Error
        error = y_pred - y_train

        # MSE
        loss = np.mean(error ** 2)

        # Manually calculate gradients
        grad_w = (2 / len(X_train)) * X_train.T @ error
        grad_b = (2 / len(X_train)) * np.sum(error)

        # Gradient descent update
        w = w - lr * grad_w
        b = b - lr * grad_b

        train_losses.append(loss)

    return w, b, train_losses