import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Load data
import pandas as pd
data = pd.read_csv('data.csv', delimiter=',', dtype=float).to_numpy()
dt = 0.1 # time step
t = np.arange(0, len(data)*dt, dt)
data = np.c_[t, data] # add time points as a new column

t = torch.from_numpy(data[:,0]).float() 
x = torch.from_numpy(data[:,1]).float() 
y = torch.from_numpy(data[:,2]).float() 


class BallDynamics(nn.Module):
    def __init__(self, g):
        super(BallDynamics, self).__init__()
        self.g = g # gravity constant
    
    def forward(self, t, z):
        x, y, vx, vy = z 
        dxdt = vx # derivative of x position
        dydt = vy # derivative of y position
        dvxdt = 0 # derivative of x velocity
        dvydt = -self.g # derivative of y velocity
        return torch.stack([dxdt, dydt, dvxdt, dvydt])

class NeuralODE(nn.Module):
    def __init__(self, hidden_size):
        super(NeuralODE, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size) 
        self.fc4 = nn.Linear(hidden_size, 4) 
        self.leaky_relu = nn.LeakyReLU() 
    
    def forward(self, t, z):
        z = self.fc1(z)
        z = self.leaky_relu(z) 
        z = self.fc2(z) 
        z = self.leaky_relu(z) 
        z = self.fc3(z) 
        z = self.leaky_relu(z) 
        z = self.fc4(z) 
        return z

def loss_fn(pred_z, true_z):
    pred_x = pred_z[:,0] 
    pred_y = pred_z[:,1] 
    true_x = true_z[:,0] 
    true_y = true_z[:,1]
    loss_x = nn.MSELoss()(pred_x, true_x)
    loss_y = nn.MSELoss()(pred_y, true_y)
    loss = loss_x + loss_y
    return loss

#initialize params
g = 9.81
hidden_size = 32 
z0 = torch.tensor([0.0, 0.0, 10.0, 10.0]) # initial condition (x0, y0, vx0, vy0)
ode_fn = BallDynamics(g) 
neural_ode_fn = NeuralODE(hidden_size)
optimizer = optim.Adam(neural_ode_fn.parameters(), lr=0.01, weight_decay=0.001) # optimizer with weight decay

epochs = 100 
for epoch in range(epochs): 
    optimizer.zero_grad()
    pred_z = odeint(neural_ode_fn, z0, t) 
    true_z = torch.stack([x, y, torch.zeros_like(x), torch.zeros_like(y)], dim=1) 
    loss = loss_fn(pred_z, true_z) 
    loss.backward() 
    optimizer.step() 
    print(f'Epoch {epoch}, Loss {loss.item()}') 

t_test = torch.linspace(0, 10, 100) 
pred_z_test = odeint(neural_ode_fn, z0, t_test) 
pred_x_test = pred_z_test[:,0].detach().numpy() 
pred_y_test = pred_z_test[:,1].detach().numpy() 
