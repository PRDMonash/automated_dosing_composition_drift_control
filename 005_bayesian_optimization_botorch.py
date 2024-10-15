import os
import torch 
import numpy as np
import os
import time
import pandas as pd
import numpy as np
import scipy.integrate as intg

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

from SF10 import SF10

# Before the start of reaction, check and update the following info: 
# 1. exp_data folder
# 2. SF10 pump port
# 3. f_target, t_i, n_iteration, initial_flow_rates, M10 and M20
# 4. Peak area integration wavelengths (MMA, BA: 1328, 1316; 2888, 2856;; MMA, BMDO: 1644, 1628; 1680, 1672)
# 5. Export folder 

#Pump Assignment and Operation
pump_stocksoln = SF10('COM14','stock solution')

# Define input
exp_data = r'C:\Users\IR112\Documents\iC IR Experiments\Export folder\Exp 2024-03-27 20-43' # Paste file path here
f_target = 0.4955 # enter the expected f
t_i = 30 + 15 
n_iteration = 500
M10 = 0.3704053274626461 # enter M10
M20 = 0.1343640062965743 # enter M20

# Define x data for training 
initial_flow_rates = torch.tensor([0, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.065])

# Define x and y tensor 
train_x = torch.tensor([])
train_y = torch.tensor([])

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

# Find peak integration of MMA, BMDO
def find_M(directory): 
    print(directory)
    print('hello Im here')
    print(f'result = {file_t(directory)}')
    M1 = peak_integration(file_t(directory), 1328, 1316)
    M2 = peak_integration(file_t(directory), 2888, 2856)
    M1t = M1/M10
    M2t = M2/M20
    return M1t, M2t

# Find latest file 
def file_t(directory): 
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# Find current f
def f_cal(M1t, M2t): 
    f = M1t/(M1t + M2t)
    return f

# Define objective function
def objective_function(target,train_Y):
    current_f = f_cal()
    #Append the measured mean intensity to current set
    train_Y = torch.cat([train_Y, torch.tensor([[current_f]])], dim=0)
    #Calculate the current score based on the most recent measurement and return score as well as whole data set as torch tensor
    difference=-abs(target-current_f) 
    return torch.tensor([difference]),train_Y


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

# To get initial train_x and train_y with initial_flow_rates
for n in range(10):
    train_x = initial_flow_rates[n]
    train_x = torch.cat((train_x, torch.tensor([initial_flow_rates[n]]).unsqueeze(-1)), dim=0).unsqueenze(-1)
    pump_stocksoln.start()
    pump_stocksoln.changeFlowrate(train_x.numpy()[0])

    time.sleep(t_i)
    
    latest_file = file_t(exp_data)
    M1t, M2t = find_M(exp_data)
    new_loss = -abs(f_cal(M1t, M2t) - f_target)
    train_y = torch.cat((train_y, torch.tensor([[new_loss]]).unsqueeze(-1)), dim=0).unsqueenze(-1)

# Experiment starts
last_processed = None 

for i in range(n_iteration): 
    if latest_file != last_processed:
        print(f'No. of iteration: {i}')
    
        # Find the next optimal flow rate 
        next_flow_rate = get_next_point(train_x, train_y, 
                                        best_observed_value = train_y.max().item(), 
                                        bounds = torch.tensor([[0.], [0.06]]), 
                                        n_points = 1).unsqueeze(-1)
        
        if next_flow_rate < 0.2: 
            next_flow_rate = 0
        else: 
            continue
        
        train_x = torch.cat((train_x, next_flow_rate))
        print(f'The next flow rate is: {next_flow_rate.numpy()[0]}')
        
        time.sleep(t_i)
        
        M1t, M2t = find_M(exp_data)
        y_new = f_cal(M1t, M2t)
        loss_new = torch.tensor(-abs(y_new - f_target)).unsqueeze(-1)
        train_y = torch.cat((train_y, y_new))
        print(f'The current f is: {y_new}')
    
    else:
        pump_stocksoln.changeFlowrate(0)
        break  # This exits the while loop

# Convert x and y to lists and export in csv 
x_list = train_x.tolist()
y_list = train_y.tolist()
print(x_list, y_list)

df_x_y = pd.DataFrame(x_list, y_list)
df_x_y.to_csv('S:/Sci-Chem/PRD/IR 112/Lily/04_11_output/optimal flow rates vs real time f.csv', index=False) # Change floder name
print(df_x_y)
