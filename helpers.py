import numpy as np
import scipy
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


class BleParams:
    slope: float = 0.21
    intercept: float = 3.92
    sigma: float = np.sqrt(0.33)
    tx: float = 0.0
    correction: float = 2.398
    name: str = 'briers-lognormal'

ble_params = BleParams()

# GLOBAL VARIABLES
idx_scaling_factor = 0
idx_ble_weights, num_ble_weights = 1, 4
idx_ble_thresholds, num_ble_thresholds = 1+4, 3
idx_con_weights, num_con_weights = 1+4+3, 2
global_temperature = 1

true_params = np.array([0.285311,0.384, 0.151, 0.017, 0.008, 55, 8, 5, 0.6, 2.0]) 

class CovidSoftModel(nn.Module):
    def __init__(self):
        super(CovidSoftModel, self).__init__()
        empty_params = true_params + abs(np.random.normal(0, 0.1, size=10))  ## todo: randomize not starting from 0
        self.params = nn.ParameterList([nn.Parameter(torch.tensor([x])) for x in empty_params])

    def forward(self, events, params=None):
      if params is None:
        return Q_j_soft_single(self.params, events)
      else:
        return Q_j_soft_single(params, events)


class CovidHardModel(nn.Module):
    def __init__(self):
        super(CovidHardModel, self).__init__()
        empty_params = true_params + abs(np.random.normal(0, 0.1, size=10))  ## todo: randomize not starting from 0
        self.params = nn.ParameterList([nn.Parameter(torch.tensor([x])) for x in empty_params])

    def forward(self, events, params=None):
      if params is None:
        return Q_j_hard_single(self.params, events)
      else:
        return Q_j_hard_single(params, events)

def distance_to_atten(distance, ble_params):
    '''
    Input:
        distance - distance between users
        ble_params - fixed constants for computing bluetooth attenuation based on distance
    Outpu:
        bluetooth attenuation based on distance
    '''
    mu = ble_params.intercept + ble_params.slope * np.log(distance)
    rssi = -np.exp(mu)
    atten = ble_params.tx - (rssi + ble_params.correction)

    return atten


def bluetooth_signal_a(d_n):
    '''
    Input:
        d_n - distances between users
    Outpu:
        bluetooth attenuations based on distances
    '''
    return distance_to_atten(d_n, ble_params)


def quantized_symptom_onset(a_n, thresholds=[-5.0,10.0,-2.0,6.0]):
    '''
    Input:
        sigma_n - time since COVID symptom onset
                  if sigma_n is in (-2, 6], high infectiousness;
                  if sigma_n is in (-5, -2] and (6, 10], standard infectiousness;
                  else, none (no infectiousness)
    Output:
        contagiousness level, one of 1 and 2 and 3
    '''
    res = torch.ones_like(a_n)  # contagiousness level is None
    res[torch.logical_and(a_n > thresholds[0], a_n  <=thresholds[1])] = 2  # contagiousness level is Standard
    res[torch.logical_and(a_n  > thresholds[2], a_n  <= thresholds[3])] = 3  # contagiousness level is High

    return res



def generate_exposure_events():
    '''
    Output: 
        np.array(events) - an array of simulated exposures generated 
                           from a fine uniform quantization of the three dimensional grid
    '''
    distances = np.linspace(0.1, 6, 80)
    durations = np.linspace(5, 60, 20)
    symptom_onset_times = np.arange(-10, 10+0.001, dtype=int)

    events = []
    for d_n in distances:
      for tau_n in durations:
          for sigma_n in symptom_onset_times:
              events.append((tau_n, d_n, sigma_n))

    return np.array(events)


def transform_events(X_prev, thresholds=[-5.0,10.0,-2.0,6.0]):  
    '''
    Input: 
        X_prev - events, each in the form (tau_n, d_n, sigma_n) = (exposure duration, distance, days symptom onset)
    Output:
        transformed events, each in the form (tau_n, a_n, c_n) = (exposure duration, bluetooth attenuation, infectiousness)
    '''
    
    X = torch.cat((X_prev[:,0].reshape(-1, 1), 
                        bluetooth_signal_a(X_prev[:,1]).reshape(-1,1), 
                        quantized_symptom_onset(X_prev[:,2], thresholds).reshape(-1,1)), axis=1)
  
    return X


def plot_iter_loss(losses):
    plt.figure(figsize=(8,5))
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss v.s # Iterations")
    plt.show()


def plot_iter_param(parameters_history):
    fig, ax = plt.subplots(5, 2, figsize=(16,25))
    indices = [(i,j) for i in range(5) for j in range(2)]

    param_hist = np.array(parameters_history)
    for k in range(10):
        diffs = param_hist[:,k]
        ax[indices[k]].plot(range(len(diffs)), diffs)
        ax[indices[k]].set_title(f"{param_names[k]}")
        ax[indices[k]].set_xlabel("Iteration")
        ax[indices[k]].set_ylabel("Estimated Parameter")
    
    plt.show()

param_names = ["Scaling Factor", 
               "Bluetooth Attenuation Weight 1", "Bluetooth Attenuation Weight 2",
               "Bluetooth Attenuation Weight 3", "Bluetooth Attenuation Weight 4",
               "Bluetooth Attenuation Threshold 1", "Bluetooth Attenuation Threshold 2", 
               "Bluetooth Attenuation Threshold 3", 
               "Infectiousness Weight 1", "Infectiousness Weight 2"]

def plot_iter_paramdiff(parameters_history):
    fig, ax = plt.subplots(5, 2, figsize=(16,25))
    indices = [(i,j) for i in range(5) for j in range(2)]

    param_hist = np.array(parameters_history)
    for k in range(10):
        diffs = param_hist[:,k] - true_params[k]
        ax[indices[k]].plot(range(len(diffs)), diffs)
        ax[indices[k]].set_title(f"{param_names[k]}")
        ax[indices[k]].set_xlabel("Iteration")
        ax[indices[k]].set_ylabel("Difference between Estimated & True Parameter")
    
    plt.show()

FPR_rate = lambda mat: mat[0,1] / (mat[0,1] + mat[0,0])
FNR_rate = lambda mat: mat[1,0] / (mat[1,0] + mat[1,1])

# SIGMOID FUNCTION
sigmoid = lambda x, temp: 1 / (1 + torch.exp(-temp*x))


# WEIGHTS FOR EACH BLUETOOTH ATTENUATION WEIGHTS USING SOFT BINNING
def tau_nb(params, x_n, b, temp):
    '''
    Input: 
        params - [mu (1), attenuation weights (4), attenuation threshold in residual form (3), infectiousness weights (2)]
        x_n - (tau_n, a_n, c_n)
              tau_n - exposure duration for all k micro exposures in one exposure
              a_n - bluetooth attenuation during exposure (alternative of distance during exposure)
              c_n - infectiousness level (alternative of time since COVID symptom onset)
        b - the bucket, should be one of 0 or 1 or 2 or 3
        temp - temperature for sigmoid
    Output:
        the soft weights for each bucket to use for a user
    '''
    tau_n, a_n, c_n = x_n[:, 0], x_n[:, 1], x_n[:, 2]
    threshold_ble = params[idx_ble_thresholds:(idx_ble_thresholds+num_ble_thresholds)]
    threshold_ble_abs = [threshold_ble[0], sum(threshold_ble[0:2]), sum(threshold_ble[0:3])]  # residual parameters to scoring parameters
    
    if b == 0:
        soft_dur = tau_n * 1 * sigmoid(threshold_ble_abs[b]-a_n, temp)
    elif b == 3:
        soft_dur = tau_n * sigmoid(a_n-threshold_ble_abs[b-1], temp) * 1
    else:
        soft_dur = tau_n * sigmoid(a_n-threshold_ble_abs[b-1], temp) * sigmoid(threshold_ble_abs[b]-a_n, temp)

    return soft_dur


# RISK SCORE USING HARD BINNING
def r_n_hard(params, x_n):
    '''
    Input:
        params - [mu (1), attenuation weights (4), attenuation threshold in residual form (3), infectiousness weights (2)]
        x_n - (tau_n, a_n, c_n)
              tau_n - exposure duration for all k micro exposures in one exposure
              a_n - bluetooth attenuation during exposure (alternative of distance during exposure)
              c_n - infectiousness level (alternative of time since COVID symptom onset)
        temp - temperature for sigmoid
    Output:
        an approximation of harzard score for exposure using hard binning
    '''
    tau_n, a_n, c_n = x_n[:,0], x_n[:,1], x_n[:,2]
    score = torch.zeros_like(a_n)

    score += tau_n
    score *= f_ble(params, x_n)
    score *= f_con(params, x_n)

    return score


# RISK SCORE USING SOFT BINNING
def r_n_soft(params, x_n, temp):
    '''
    Input:
        params - [mu (1), attenuation weights (4), attenuation threshold in residual form (3), infectiousness weights (2)]
        x_n - (tau_n, a_n, c_n)
              tau_n - exposure duration for all k micro exposures in one exposure
              a_n - bluetooth attenuation during exposure (alternative of distance during exposure)
              c_n - infectiousness level (alternative of time since COVID symptom onset)
        temp - temperature for sigmoid
    Output:
        an approximation of harzard score for exposure using soft binning
    '''
    tau_n, a_n, c_n = x_n[:,0], x_n[:,1], x_n[:,2]
    score = torch.zeros_like(a_n)
    weight_ble = params[idx_ble_weights:(idx_ble_weights+num_ble_weights)]  # params[1:(1+4)]

    for ble_bucket in range(4):
        score += tau_nb(params, x_n, ble_bucket, temp) * weight_ble[ble_bucket]

    score *= f_con(params, x_n)

    return score


# INFECTION PROBABILITY USING HARD BINNING
def Q_j_hard_single(params, events):
    '''
    Input: 
        params - [mu (1), attenuation weights (4), attenuation threshold in residual form (3), infectiousness weights (2)]
        events - each entry in the form (tau_n, a_n, c_n)
                 tau_n - exposure duration for all k micro exposures in one exposure
                 a_n - bluetooth attenuation during exposure (alternative of distance during exposure)
                 c_n - infectiousness level (alternative of time since COVID symptom onset)
    Output:
        probabilities of infection for multiple events using hard binning
    '''
    mu = params[idx_scaling_factor]
    r_n = r_n_hard(params, events)  # risk score

    return 1-torch.exp(-mu*r_n)

# SUB RISK SCORE FOR DAYS SINCE SYMPTOM ONSET BY USING INFECTIOUSNESS
def f_con(params, x_n):
    '''
    Input:
        params - [mu (1), attenuation weights (4), attenuation threshold in residual form (3), infectiousness weights (2)]
        x_n - (tau_n, a_n, c_n)
              tau_n - exposure duration for all k micro exposures in one exposure
              a_n - bluetooth attenuation during exposure (alternative of distance during exposure)
              c_n - infectiousness level (alternative of time since COVID symptom onset)
    Output: 
        an approximation of simulated risk given time since COVID symptom onset using infectiousness
    '''
    c_n = x_n[:,2] 
    weight_con = params[idx_con_weights:]  # params[(1+4+3):]  

    res = torch.zeros_like(c_n)
    res[c_n == 2] = weight_con[0]
    res[c_n == 3] = weight_con[1]

    return res

# PREVENT PARAMETERS FROM GETTING TOO EXTREME IN TRAINING
def project(params): 
    params[idx_scaling_factor].data.clamp_(1e-8, 1.)
    params[idx_ble_weights:(idx_ble_weights+num_ble_weights)].data.clamp_(1e-6, 1e3)
    params[idx_ble_thresholds].data.clamp_(10, 60)
    params[(idx_ble_thresholds+1):(idx_ble_thresholds+num_ble_thresholds)].data.clamp_(1, 25)
    params[idx_con_weights:].data.clamp_(1e-6, 1e3)


def gen_label(ys):
    p = np.random.uniform(0, 1, len(ys))
    res = np.zeros(len(ys))
    res = (p < ys.detach().numpy()).astype(int)

    return torch.tensor(res)

# SUB RISK SCORE FOR DISTANCE BY USING BLUETOOTH ATTENUATION
def f_ble(params, x_n):
    '''
    Input:
        params - [mu (1), attenuation weights (4), attenuation threshold in residual form (3), infectiousness weights (2)]
        x_n - (tau_n, a_n, c_n)
              tau_n - exposure duration # for all k micro exposures in one exposure (not considering micro exposures for now)
              a_n - bluetooth attenuation during exposure (alternative of distance during exposure)
              c_n - infectiousness level (alternative of time since COVID symptom onset)
    Output:
        an approximation of simulated risk given distance using bluetooth attenuation
    '''
    a_n = x_n[:,1]
    res = torch.zeros_like(a_n)

    threshold_ble = params[idx_ble_thresholds:(idx_ble_thresholds+num_ble_thresholds)]  # params[(1+4):(1+4+3)]
    threshold_ble_abs = [threshold_ble[0], sum(threshold_ble[0:2]), sum(threshold_ble[0:3])]  # residual parameters to scoring parameters
    weight_ble = params[idx_ble_weights:(idx_ble_weights+num_ble_weights)]  # params[1:(1+4)]

    res[a_n <= threshold_ble_abs[0]] = weight_ble[0]
    res[torch.logical_and(a_n > threshold_ble_abs[0] , a_n <= threshold_ble_abs[1])] = weight_ble[1]
    res[torch.logical_and(a_n > threshold_ble_abs[1] , a_n <= threshold_ble_abs[2])] = weight_ble[2]
    res[a_n > threshold_ble_abs[2]] = weight_ble[3]

    return res


# INFECTION PROBABILITY USING SOFT BINNING
def Q_j_soft_single(params, events, global_temperature = 5):
    '''
    Input: 
        params - [mu (1), attenuation weights (4), attenuation threshold in residual form (3), infectiousness weights (2)]
        events - each entry in the form (tau_n, a_n, c_n)
                 tau_n - exposure duration for all k micro exposures in one exposure
                 a_n - bluetooth attenuation during exposure (alternative of distance during exposure)
                 c_n - infectiousness level (alternative of time since COVID symptom onset)
    Output:
        probabilities of infection for multiple events using soft binning
    '''
    mu = params[idx_scaling_factor]
    r_n = r_n_soft(params, events, global_temperature)  # risk score

    return 1-torch.exp(-mu*r_n)


def soft_hard_surfaces(params):
    fig = plt.figure(figsize=(16,8))

    xx = np.linspace(-10, 10.0001, 21)  # days since symptom onset
    yy = bluetooth_signal_a(np.linspace(0.1, 5, 20))  # bluetooch attenuation
    yy, xx= np.meshgrid(yy, xx)

    Z_display_soft = []
    Z_display_hard = []
    X_input = []
    for subx, suby in zip(xx.flatten(), yy.flatten()):
        X_input.append([15, suby, quantized_symptom_onset(torch.tensor([subx]))[0]])

    X_input_tensor = torch.tensor(np.array(X_input))
    Z_display_soft = CovidSoftModel().forward(X_input_tensor, params)
    Z_display_hard = CovidHardModel().forward(X_input_tensor, params)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_trisurf(xx.flatten(), yy.flatten(), Z_display_soft.detach().numpy().flatten())
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooch Attenuation')
    ax.set_zlabel('Infection Probability')
    ax.set_title("Approximate Risk Surface by the Model, SOFT BINNING")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_trisurf(xx.flatten(), yy.flatten(), Z_display_hard.detach().numpy().flatten())
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooch Attenuation')
    ax.set_zlabel('Infection Probability')
    ax.set_title("Approximate Risk Surface by the Model, HARD BINNING")

    plt.show()

def risk_surfaces(true_params, init_params, current_params, binning='soft'):
    fig = plt.figure(figsize=(18,6))

    xx = np.linspace(-10, 10.0001, 21)  # days since symptom onset
    yy = bluetooth_signal_a(np.linspace(0.1, 5, 20))  # bluetooch attenuation
    yy, xx= np.meshgrid(yy, xx) 

    Z_display_true, Z_display_init, Z_display_current = [], [], []
    X_input = []
    for subx, suby in zip(xx.flatten(), yy.flatten()):
        X_input.append([15, suby, quantized_symptom_onset(torch.tensor([subx]))[0]])
    
    X_input_tensor = torch.tensor(np.array(X_input))
    if binning == 'soft':
        Z_display_true = CovidSoftModel().forward(X_input_tensor, true_params)
        Z_display_init = CovidSoftModel().forward(X_input_tensor, init_params)
        Z_display_current = CovidSoftModel().forward(X_input_tensor, current_params)
    elif binning == 'hard':
        Z_display_true = CovidHardModel().forward(X_input_tensor, true_params)
        Z_display_init = CovidHardModel().forward(X_input_tensor, init_params)
        Z_display_current = CovidHardModel().forward(X_input_tensor, current_params)

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_trisurf(xx.flatten(), yy.flatten(), Z_display_true.detach().numpy().flatten())
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooch Attenuation')
    ax.set_zlabel('Infection Probability')
    ax.set_title(f"Risk Surface, True Parameters, {binning.upper()} BINNING")

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_trisurf(xx.flatten(), yy.flatten(), Z_display_init.detach().numpy().flatten())
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooch Attenuation')
    ax.set_zlabel('Infection Probability')
    ax.set_title(f"Risk Surface, Random Initial Parameters, {binning.upper()} BINNING")

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot_trisurf(xx.flatten(), yy.flatten(), Z_display_current.detach().numpy().flatten())
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooch Attenuation')
    ax.set_zlabel('Infection Probability')
    ax.set_title(f"Risk Surface, Estimated Parameters, {binning.upper()} BINNING")

    plt.show()

def compare_surfaces(params):
    fig = plt.figure(figsize=(16,8))

    xxd = np.linspace(-10, 10.0001, 21)  # days since symptom onset
    yyd = np.linspace(0.1, 5, 20)  # distance
    yyd, xxd= np.meshgrid(yyd, xxd)

    xxb = np.linspace(-10, 10.0001, 21)  # days since symptom onset
    yyb = bluetooth_signal_a(np.linspace(0.1, 5, 20))  # bluetooch attenuation
    yyb, xxb= np.meshgrid(yyb, xxb)

    X_input_d = []
    for subx, suby in zip(xxd.flatten(), yyd.flatten()):
        X_input_d.append([15, suby, subx])
    X_input_d = np.array(X_input_d)
    Z_display_true = pn(X_input_d)

    Z_display_soft = []
    X_input_soft = []
    for subx, suby in zip(xxb.flatten(), yyb.flatten()):
        X_input_soft.append([15, suby, quantized_symptom_onset(np.array([subx]))[0]])
    X_input_soft_tensor = torch.tensor(np.array(X_input_soft))
    Z_display_soft = CovidSoftModel().forward(X_input_soft_tensor, params)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_trisurf(xxd.flatten(), yyd.flatten(), Z_display_true.flatten(), alpha=0.4)
    ax.plot_trisurf(xxd.flatten(), yyd.flatten(), Z_display_soft.detach().numpy().flatten(), cmap='copper')
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Infection Probability')
    ax.set_title("Risk Surfaces, Duration = 15 min")

    # TODO: should we delete the block below?
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.contourf(xxb, yyb, Z_display_true.reshape(21,20), 100)
    ax.contourf(xxb, yyb, Z_display_soft.detach().numpy().reshape(21,20), 100, cmap='copper')
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooch Attenuation')
    ax.set_zlabel('Infection Probability')
    ax.set_title("Risk Surfaces, Duration = 15 min")

def plot_iter_loss_boot(losses_allboot):
    losses_ub = np.percentile(losses_allboot, 97.5, axis=0)  # 97.5 th percentile
    losses_mean = np.percentile(losses_allboot, 50.0, axis=0)  # mean
    losses_lb = np.percentile(losses_allboot, 2.5, axis=0)  # 2.5 percentile

    plt.figure(figsize=(8,5))
    plt.plot(range(len(losses_mean)), losses_mean, 'b-', label='Mean Loss')
    plt.fill_between(range(len(losses_mean)), losses_ub, losses_lb, color='blue', alpha=0.2, label='95% Interval for Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss v.s # Iterations")
    plt.legend()
    plt.show()

def plot_param_distribution(params_current_allboot):
    fig, ax = plt.subplots(5, 2, figsize=(16,25))
    indices = [(i,j) for i in range(5) for j in range(2)]

    for k in range(10):
        paramk_mean = np.mean(params_current_allboot[:,k])
        sns.kdeplot(params_current_allboot[:,k], ax=ax[indices[k]])
        ax[indices[k]].set_xlabel("Parameter Value")
        ymax = ax[indices[k]].get_yticks()[-1]
        ax[indices[k]].vlines(paramk_mean, 0, ymax, colors="red", linestyles="dashed", label="Mean Parameter Value")
        ax[indices[k]].vlines(true_params[k], 0, ymax, colors="black", linestyles="dashed", label="True Parameter Value")
        ax[indices[k]].set_title(f"{param_names[k]}: KDE plot")
        ax[indices[k]].legend()


def metric_boot(y_preds_proba_boot, y_labels_boot, num_boot=100):
    accs_boot, FPRs_boot, FNRs_boot = [], [], []
    for i in range(num_boot):
        cmat_boot = confusion_matrix(y_labels_boot[i], (y_preds_proba_boot.data > 0.5).int())
        accs_boot.append(accuracy_score(y_labels_boot[i], (y_preds_proba_boot.data > 0.5).int()))
        FPRs_boot.append(FPR_rate(cmat_boot))
        FNRs_boot.append(FNR_rate(cmat_boot))
    
    return np.array(accs_boot), np.array(FPRs_boot), np.array(FNRs_boot)


# MODEL TRAINING FUNCTION
## TODO: correct we did not use BCE, we used MSE
def train_model_two_steps_and_temperature(X, y, params_init=None, num_iters=2000, lr=0.001, batch_size=300, momentum=0.9, loss_bce=False):
    

    temperature_values = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5,5,5,5,5]
    global_temperature = temperature_values[0]

    # TODO: ADD BATCH SIZE, DO SGD
    if params_init is None:
        params_init = true_params + abs(np.random.normal(0, 0.1, size=10))  # change to RANDOM NUMBERS later

    params_current = torch.tensor(params_init, requires_grad=True)
    parameters_history = [np.array(params_current.data)]
    if loss_bce:
      loss_fn = torch.nn.BCELoss()
    else:
      loss_fn = torch.nn.MSELoss()

    losses = [loss_fn(Q_j_soft_single(params_current, X, global_temperature).float(), y.float())]

    for i in range(num_iters):  
        curr_loss = losses[-1]
        if i % 100 == 0:
            lr *= momentum  # decaying learning rate
            print(f"Epoch {i} loss: {curr_loss}")
            #print(f"Current gradient: {grad}")
            #print(f"Current parameters: {params_current.data}")

        loss = loss_fn(Q_j_soft_single(params_current, X, global_temperature).float(), y.float())
        loss.backward()
        grad = params_current.grad.data

        params_current.data[idx_ble_thresholds:(idx_ble_thresholds+num_ble_thresholds)] -= lr * grad[idx_ble_thresholds:(idx_ble_thresholds+num_ble_thresholds)]
        params_current.data[(idx_ble_thresholds+num_ble_thresholds):] -= 0.1 * lr * grad[(idx_ble_thresholds+num_ble_thresholds):]
        params_current.data[:idx_ble_thresholds] -= 0.1 * lr * grad[:idx_ble_thresholds]
        
        project(params_current)
        parameters_history.append(np.array(params_current.data))

        curr_loss = loss.detach() 
        losses.append(curr_loss)

        if i % (num_iters // len(temperature_values)) == 0:
          global_temperature = temperature_values[i//(num_iters //len(temperature_values))]
          print(f"Global temperature is {global_temperature}")
    
    y_preds_proba = Q_j_soft_single(params_current, X)

    return params_current.data, parameters_history, losses, y_preds_proba.data

def train_model_with_thresholds(X, y, thresholds, num_iters=1000, lr=0.001, momentum=0.9):

    soft_model = CovidSoftModel()
    optimizer = optim.SGD(soft_model.parameters(), lr=0.001, momentum=0.9)

    parameters_history = []

    loss_fn = torch.nn.MSELoss()

    losses = []

    X_approx = transform_events(X, thresholds)

    for i in range(num_iters):  
        optimizer.zero_grad()

        output = soft_model(X_approx).double()
  
        loss = loss_fn(output, y)
   
        losses.append(loss.detach())

        curr_params = []
        for name, param in soft_model.named_parameters():
            if param.requires_grad: 
                curr_params.append(param.data)
                #print(name, param.grad)

        curr_params = torch.tensor(np.array(curr_params))
        #project_with_thresholds(curr_params)
        parameters_history.append(np.array(curr_params))

        loss.backward(retain_graph=True)

        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {i} loss: {loss}")
    
    y_preds_proba = soft_model(X_approx)

    return soft_model.parameters(), np.array(parameters_history), losses, y_preds_proba.data

def print_results_fixscale(exs, y_train): 
    '''
        take a list of experiments
    '''

    y_train= (y_train > 0.5).int()
    if len(exs) == 1:
        ex = exs[0]

        cmat = confusion_matrix(y_train, (ex[0][3].data > 0.5).int())
        acc = accuracy_score(y_train, (ex[0][3].data > 0.5).int())
        f1 = f1_score(y_train, (ex[0][3].data > 0.5).int())

        return [float(ex[0][2][-1]),acc,f1]

    if len(exs) > 1:
        results = []
        
        for ex in exs:

            cmat = confusion_matrix(y_train, (ex[0][3].data > 0.5).int())
            acc = accuracy_score(y_train, (ex[0][3].data > 0.5).int())
            f1 = f1_score(y_train, (ex[0][3].data > 0.5).int())

            results.append([float(ex[0][2][-1]),acc,f1])
        
        return results

# GLOBAL VARIABLES
# Auxiliary functions for true model
D_min = 1
mu = -4
sigma = 1.85
alpha = 5.85
tau = 5.42
# lbda = 3.1e-6 # param in the paper
lbda = 1.1e-1 

def f_dist(d, D_min=1):
    '''
    Input:
        d - distance between people during exposure
        D_min - a constant used for truancation
    Output:
        simulated risk given distance
    '''
    return np.clip(D_min**2/d**2, a_min = None, a_max=1)

def f_inf(sigma_n, mu=-4, sigma=1.85, alpha=5.85, tau=5.42):
    '''
    Input:
        sigma_n - time since COVID symptom onset
        mu, sigma, alpha, tau - parameters for the Skewed Logistic distribution
        here, we simplify the equation by setting t_i (the incubation time) = tau
    Output:
        simulated risk probability given time since COVID symptom onset
    '''
    return infectiousness_skew_logistic(sigma_n)

def incubation_dist(t):
    mu = 1.621
    sig = 0.418
    rv = scipy.stats.lognorm(sig, scale=np.exp(mu))
    return rv.pdf(t)

# Symptom days to infectiousness
def skew_logistic_scaled(x, alpha, mu, sigma):
    return scipy.stats.genlogistic.pdf(x, alpha, loc=mu, scale=sigma)

def ptost_conditional(ts, incubation):
    mu = -4
    sigma = 1.85
    alpha = 5.85
    tau = 5.42
    fpos = skew_logistic_scaled(ts, alpha, mu, sigma)
    fneg = skew_logistic_scaled(ts*tau/incubation, alpha, mu, sigma)
    ps = fpos
    neg = np.where(ts < 0)
    ps[neg] = fneg[neg]
    ps = ps/np.max(ps)
    return ps

def ptost_uncond(tost_times):
    incub_times = np.arange(1, 14, 1)
    incub_probs = incubation_dist(incub_times) 
    tost_probs = np.zeros_like(tost_times, dtype=float)
    for k, incub in enumerate(incub_times):
        ps = ptost_conditional(tost_times, incub)
        tost_probs += incub_probs[k] * ps
    return tost_probs

infectiousness_curve_times = np.arange(-14, 14+1, 0.1)
infectiousness_curve_vals = ptost_uncond(infectiousness_curve_times)

def infectiousness_skew_logistic(delta):
    return np.interp(delta, infectiousness_curve_times, infectiousness_curve_vals)

def sn(tau_n, d_n, sigma_n):
    '''
    Input:
        tau_n - exposure duration
        d_n - distance during exposure
        sigma_n - time since COVID symptom onset
    Output:
        harzard score for exposure
    '''

    return tau_n*f_dist(d_n, D_min)*f_inf(sigma_n, mu, sigma, alpha, tau)


def pn(x_n):
    '''
    Input:
        x_n - wrapped up (exposure duration, distance during exposure, time since COVID symptom onset)
    Output:
        infection probability given single exposure
    '''
  
    tau_n, d_n, sigma_n = x_n[:,[0]], x_n[:,[1]], x_n[:,[2]]
    return 1-np.exp(-lbda*sn(tau_n, d_n, sigma_n))


class CovidTrueModel():
    def forward(self, x):
        # Perform forward pass and return labels
        probs = pn(x)
        return self.gen_label(np.array(probs.reshape(-1)))

    def get_probs(self, x):
        probs = pn(x)
        return probs.reshape(-1)
        
    def gen_label(self, ys):
        p = np.random.uniform(0,1, len(ys))
        res = np.zeros(len(ys))
        res = (p < ys).astype(int)

        return res

    def get_events(self):
        return generate_exposure_events()

def plot_superposed(params, thresholds =[-5.0,10.0,-2.0,6.0], t=15):
    Z_display = []
    X_input = []
    xs = []
    ys = []
    for x in np.linspace(0.1, 5, 80):
        for y in np.linspace(-10,10,21):
            xs.append(x)
            ys.append(y)
            X_input.append([t, x, y])

    xs = np.array(xs)
    ys = np.array(ys)
    X_input = np.array(X_input)

    Z_display = CovidTrueModel().get_probs(torch.tensor(X_input)).reshape(80,21)

    X_input = []
    xs = []
    ys = []
    for x in bluetooth_signal_a(torch.tensor(np.linspace(0.1, 5, 80))):
        for y in np.linspace(-10,10,21):
            xs.append(x)
            ys.append(y)
            X_input.append([t, x, quantized_symptom_onset(torch.tensor([y]), thresholds=thresholds)])

    xs = np.array(xs)
    ys = np.array(ys)
    X_input = torch.tensor(X_input)

    Z_display_approx = CovidSoftModel().forward(X_input, params).reshape(80,21)

    xs = xs.reshape(80,21)
    ys = ys.reshape(80,21)

    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.set_title("Superposition of target (blue) and learned model (copper)")
    ax.plot_trisurf(ys.T.flatten(),xs.T.flatten(), Z_display.T.flatten(), alpha=0.4)
    ax.plot_trisurf(ys.T.flatten(),xs.T.flatten(), Z_display_approx.T.flatten(), cmap="copper")
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooth Attenuation')
    ax.set_zlabel('Infection Probability');


def plot_true_model(t = 15):
    Z_display = []
    X_input = []
    xs = []
    ys = []
    for x in np.linspace(0.1, 5, 80):
        for y in np.linspace(-10,10,21):
            xs.append(x)
            ys.append(y)
            X_input.append([t, x, y])

    xs = bluetooth_signal_a(torch.tensor(xs))
    ys = np.array(ys)
    X_input = np.array(X_input)

    Z_display = CovidTrueModel().get_probs(torch.tensor(X_input)).reshape(80,21)
    xs = xs.reshape(80,21)
    ys = ys.reshape(80,21)

    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(ys.T.flatten(),xs.T.flatten(), Z_display.T.flatten())
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooch Attenuation')
    ax.set_zlabel('Infection Probability')
    ax.set_title("Target solution")

def plot_comparison_with_true(parameters_history):
    plot_true_model()
    plot_soft_model("Initial solution", torch.tensor(parameters_history[0]))
    plot_soft_model("Optimized solution", torch.tensor(parameters_history[-1]))


def plot_soft_model(title, params):
    xx = bluetooth_signal_a(np.linspace(0.1, 5, 20))# distance
    yy = np.linspace(-10, 10.0001, 21) # sigma
    yy,xx= np.meshgrid(yy, xx)


    # idx = np.where(abs(X[:,0] - 16.5789) < 0.1)[0]
    Z_display = []
    X_input = []
    for subx, suby in zip(xx.flatten(), yy.flatten()):
        X_input.append([15, subx, quantized_symptom_onset(torch.tensor([suby]))])

    tensor = torch.tensor(X_input)
    Z_display = CovidSoftModel().forward(tensor, params)

    # print(np.where(X[:,0]==15.0))
    #Z_display = y[idx]

    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    # ax.contour3D(yy.T, xx.T, Z_display.detach().numpy().reshape(80,21), 50)
    ax.plot_trisurf(yy.flatten(),xx.flatten(), Z_display.detach().numpy().flatten())
    ax.set_xlabel('Symptom Onset')
    ax.set_ylabel('Bluetooch Attenuation')
    ax.set_zlabel('Infection Probability')
    ax.set_title(title)