import pandas as pd
import numpy as np

gravity = 9.8  
initial_velocity = 50 
bounce_loss = 0.8 
num_bounces = 0 
max_bounces = 5  # maximum number of bounces before the ball is kicked again

total_time = 10  
time = np.linspace(0, total_time, 600)  # divide total time into 600 intervals
dt = time[1] - time[0] 

x = np.zeros_like(time)
y = np.zeros_like(time)
angle = np.random.uniform(-60, 60)
v_x = initial_velocity * np.cos(np.radians(angle))
v_y = initial_velocity * np.sin(np.radians(angle))

for i in range(1, len(time)):
    x[i] = x[i-1] + v_x * dt
    y[i] = y[i-1] + v_y * dt

    v_y -= gravity * dt
    if y[i] < 0:
        y[i] = 0
        v_y = -v_y * bounce_loss
        num_bounces += 1

    if x[i] > 100:
        x[i] = 100
        v_x = -v_x * bounce_loss
    if num_bounces >= max_bounces:
        angle = np.random.uniform(-60, 60)  
        v_x = initial_velocity * np.cos(np.radians(angle))
        v_y = initial_velocity * np.sin(np.radians(angle))
        num_bounces = 0

df = pd.DataFrame({'x_position': x, 'y_position': y})
df.to_csv('ball_positions.csv', index=False)
