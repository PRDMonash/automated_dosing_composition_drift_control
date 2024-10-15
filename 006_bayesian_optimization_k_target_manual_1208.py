# READ ME: 
# 1. exp_data folder 
# 2. SF10 pump port 
# 3. f_target, t_i, n_iteration, initial_flow_rates 
# 4. M10 and M20 
# 5. Peak area integration wavelengths (MMA, BA: 1328, 1316; 2888, 2856;; MMA, BMDO: 1644, 1628; 1680, 1672) 
# 6. Export folder 

import os
import torch 
import numpy as np
import os
import time
import pandas as pd
import numpy as np
import scipy.integrate as intg
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

from SF10 import SF10

#Pump Assignment and Operation
pump_stocksoln = SF10('COM8','stock solution')

# Define input
exp_data = r'C:\Users\IR112\Documents\iC IR Experiments\Export folder\Exp 2024-08-08 10-35' # Paste file path here
f_initial = 0.5 # enter the initial f
k_target = 0.002
t_i = 120 # enter the expected calculation rate 
t_ir = 30 # the time for reactor content to reach the ir probe 
t_sampling = 60
# t_ir_total = t_ir + 60
t_m = (t_i+ t_ir + t_sampling/2)/60
n_iteration = 500
M10 = 0.2568066737575286 # enter M10
M20 = 0.8202187768373108 # enter M20

n_MMA_0 = 0.01395
n_BMDO_0 = 0.01395

# Define x data for training 
initial_flow_rates = [0, 0.02, 0.025, 0.03]

# Define x and y tensor 
train_x = torch.tensor([]).unsqueeze(-1)
train_y = torch.tensor([]).unsqueeze(-1)

# Peak integration function 
def peak_integration(file,A,B):
    with open(file) as i:
        column_names = ['absorption']
        df = pd.read_csv(i,skiprows=1,index_col=0,names=column_names)
        data_df = df.loc[A:B]
#          print(data_df)
        
        #select wavenumber range for integration
        x_data = data_df.index
        y_data = data_df['absorption'].values
#         print(f'{x_data=}, {y_data=}')
        
        #baseline
        y_base1 = data_df.iloc[0,0]
        y_base2 = data_df.iloc[-1,0]
#         print(f'{y_base1=}, {y_base2=}')
        
        #Peak integration with Trapezoidal rule
        #Total Area
        peak_area1 = abs(intg.trapezoid(y=y_data,x=x_data)) #multiply by -1 because the wavenumber is in descending order

        #Area of baseline
        peak_area2 = abs(intg.trapezoid(y=[y_base1,y_base2],x=[A,B]))
        peak_area = peak_area1 - peak_area2
#         print(f'{peak_area=}, {peak_area1=}, {peak_area2=}')
        
    return peak_area

def file_t(directory): 
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# Find latest n files
def file_t(directory, n): 
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files.sort()
    # latest_file = max(files, key=os.path.getctime)
    print(files[-n-1:])
    return files[-n-1:]

# Find peak integration of MMA, BMDO
def find_M(directory, n): 
    # print(directory, n)
    files = file_t(directory, n)
    print(files)
    M1_values, M2_values = [], []

    for file in files: 
        M1 = peak_integration(file, 1328, 1316)
        M2 = peak_integration(file, 1412, 1396)
        M1t = M1/M10*n_MMA_0
        M2t = M2/M20*n_BMDO_0
        M1_values.append(M1t)
        M2_values.append(M2t)
    
    mean_M1t = np.mean(M1_values)
    mean_M2t = np.mean(M2_values)
    
    # print('hello Im here')
    # print(f'result = {file_t(directory)}')

    return mean_M1t, mean_M2t

# Find current f
def f_cal(M1t, M2t): 
    f = M1t/(M1t + M2t)
    return f

def latest_f(directory, n): 
    M1t, M2t = find_M(directory, n)
    f = f_cal(M1t, M2t)
    return f

# Function to get next optimal flow rate based on Bayesian Optimization with Botorch Monte Carlo Expected Improvement
def get_next_point(train_x, train_y, best_observed_value, bounds, n_points=1): 
    single_model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_model(mll)
    
    EI = qExpectedImprovement(model=single_model, best_f=best_observed_value)

    candidates, _ = optimize_acqf(
                    acq_function=EI,
                    bounds=bounds,
                    q=n_points,
                    num_restarts=200,
                    raw_samples=512,
                    options={"batch_limit": 5, "maxiter": 200})
    return candidates

first_f = latest_f(exp_data, 10)

f_last = f_initial

time_list = [0]
flow_rate_output = [0, 0.02, 0.025, 0.03]
time_output = [0]
latest_f_output = [first_f]
current_k_output = []

t_current = 0

# To get initial train_x and train_y with initial_flow_rates
for n in range(4):
    # Add initial flow rates to train x tensor
    train_x = torch.cat((train_x, torch.tensor([initial_flow_rates[n]]).unsqueeze(-1)))

    # Start pumping at #n initial flow rates for t_i seconds 
    pump_stocksoln.start()
    pump_stocksoln.changeFlowrate(float(initial_flow_rates[n]))
    time.sleep(t_i)

    # Pause for t_ir seconds for reactor content to reach IR probe 
    pump_stocksoln.changeFlowrate(0)
    time.sleep(t_ir + t_sampling)

    # t_current = t_cal(n)

    # Calculate the latest f value based on the mean M1t and M2t retrieved from the latest 6 files 
    f_latest = latest_f(exp_data, 10)
    
    # Calculate the current t by adding t_i and half of t_ir 
    t_current = 3 + (n)*(t_i + t_ir + t_sampling/2)/60

    print(f'latest f: {f_latest}, current time: {t_current} mins')

    # # Append latest f to y_output for linear regression and data output 
    latest_f_output.append(f_latest)
    # if len(latest_f_output) == 5: 
    #     latest_f_output.pop(0)

    # # Append t current to time output 
    # time_list.append(t_current)
    time_output.append(t_current)
    # if len(time_output) == 5: 
    #     time_output.pop(0)
    
    # Create a df for linear regression 
    plot_df = pd.DataFrame({'time': time_output, 'f_output': latest_f_output})

    plot_time = plot_df['time'].values.reshape(-1,1)
    plot_f = plot_df['f_output'].values

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(plot_time, plot_f)

    # Calculate k by fitting data with linear regression model 
    k_linear_regression = model.coef_[0]
    b_constant = model.intercept_

    print(f"This is modeling iteration #{n+1}. Fitted line around peak: y = {k_linear_regression} * time + {b_constant}")

    # Calculate the current k (to check if this method can give precise result of current k)
    k_current = (f_latest - f_last)/t_m
    print(f"This is modeling iteration #{n+1}. current k = {k_current}")

    # Append k current to k output 
    current_k_output.append(k_current)

    # Update Bayesian Optimization with new loss
    new_loss = -abs(k_current - k_target)
    train_y = torch.cat((train_y, torch.tensor([new_loss]).unsqueeze(-1)))

    # print(f'manual k calculation gives k = {k_current_calculation}')
    f_last = f_latest

    # print(train_y)

train_x = train_x.to(torch.double)
train_y = train_y.to(torch.double)

# To control slope as a predetermined constant k using Bayesian Optimization 

last_processed = None 

for i in range(n_iteration): 
    latest_file = file_t(exp_data, 1)

    # Ckech if IR is still running  
    if latest_file != last_processed:
        best_observed_value = train_y.max().item()
        # Find the next optimal flow rate 
        next_flow_rate = get_next_point(train_x, train_y, 
                                        best_observed_value, 
                                        bounds = torch.tensor([[0.], [0.1]]), 
                                        n_points = 1)
        train_x = torch.cat((train_x, next_flow_rate))

        x_next = next_flow_rate[0][0]

        # To overcome the limitation that smallest input = 0.2 
        # if x_next < 0.02: 
        #     x_next += 0.005

        flow_rate_output.append(x_next)

        # print(x_output)

        # print(f'The next flow rate is: {x_next}')
        
        pump_stocksoln.changeFlowrate(x_next)

        time.sleep(t_i)
        
        pump_stocksoln.changeFlowrate(0)
        time.sleep(t_ir + t_sampling)

        f_latest = latest_f(exp_data, 10)
        latest_f_output.append(f_latest)
        # if len(latest_f_output) == 5: 
        #     latest_f_output.pop(0)

        t_current += (t_i + t_ir + t_sampling/2)/60

        print(f'latest f: {f_latest}, current time: {t_current} mins')
        
        time_list.append(t_current)
        time_output.append(t_current)
        # if len(time_output) == 5: 
        #     time_output.pop(0)

        # Create a df for linear regression 
        plot_df = pd.DataFrame({'time': time_output, 'f_output': latest_f_output})

        plot_time = plot_df['time'].values.reshape(-1,1)
        plot_f = plot_df['f_output'].values

        # Initialize and fit the linear regression model
        model = LinearRegression()
        model.fit(plot_time, plot_f)

        # Calculate k by fitting data with linear regression model 
        k_linear_regression = model.coef_[0]
        b_constant = model.intercept_

        print(f"This is modeling iteration #{n+1}. Fitted line around peak: y = {k_linear_regression} * time + {b_constant}")

        # Manual calculation of k 
        k_current = (f_latest - f_last)/t_m 

        # print(f'k from manual calculatin is: {k_current_calculation}')
        # print(f'loss_new: {loss_new}')
        # print(f'train_y: {train_y}')

        current_k_output.append(k_current)

        new_loss = -abs(k_current - k_target)
        f_last = f_latest

        train_x = train_x.to(torch.double)
        train_y = train_y.to(torch.double)

        train_y = torch.cat((train_y, torch.tensor([new_loss]).unsqueeze(-1)))

        train_y = train_y.to(torch.double)
        train_x = train_x.to(torch.double)

        # print(f'The current f is: {y_new}')
        print(f'This is iteration # {i+4}, current f is {f_latest}, current k is {k_current}')
        last_processed = latest_file
    
    else:
        pump_stocksoln.changeFlowrate(0)
        break  # This exits the while loop

df_x_y = pd.DataFrame(data={'time_minutes': time_output, 'optimal_flow_rate':flow_rate_output, 'k_output':current_k_output})
df_x_y.to_csv('C:/Users/IR112/Desktop/000_python code/export_folder/k constant 0.002 f last linear regression 07 08 24.csv', index=False) # Change floder name
print(df_x_y)