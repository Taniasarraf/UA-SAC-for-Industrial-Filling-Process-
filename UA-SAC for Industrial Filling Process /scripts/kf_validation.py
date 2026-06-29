import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import matplotlib.font_manager as fm 

target_weight = 750.0
dt = 0.01
quantization = 10.0
coarse_rate = 600.0
fine_rate = 60.0           
sigma_proc = 5.0           
sigma_meas = 3.0           

mu_residual = 15.0
sigma_residual = 2.5


H = np.array([[1.0, 0.0, 0.0]])
Q_nominal = np.diag([1.0, 1.5, 0.05])   
R_nominal = (sigma_meas**2) + ((quantization**2) / 12.0)
inflation_window = 10


num_trials = 100
max_steps = 1000           
time_steps = np.arange(0, max_steps) * dt
n = len(time_steps)

all_nis = np.zeros((num_trials, n))
all_nees = np.zeros((num_trials, n))
all_weight_errors = np.zeros((num_trials, n))
all_flow_errors = np.zeros((num_trials, n))
final_weights = []

vis_data = {k: [] for k in ['time', 'true_w', 'kf_w', 'meas_w', 'true_dot_m', 'kf_dot_m']}

print(f"Executing Mathematically Aligned Simulation Engine ({num_trials} Trials)...")

for trial in range(num_trials):
    true_alpha = np.random.uniform(0.7, 1.5)
    x_hat = np.array([[0.0], [0.0], [1.0]])
    P = np.diag([10.0, 2.0, 0.1])     
    
    current_weight = 0.0
    is_fine = False
    inflation_counter = 0
    switch_weight = None
    
    simulated_agent_switch_threshold = np.random.uniform(430.0, 480.0)
    actual_end_idx = n - 1
    
    for k in range(n):
        if not is_fine and current_weight >= simulated_agent_switch_threshold:
            is_fine = True
            inflation_counter = inflation_window
            switch_weight = current_weight
            
        m_dot_u = fine_rate if is_fine else coarse_rate
        
        if k < 15:
            Q = np.diag([50.0, 250.0, 10.0])
            R_k = R_nominal * 10.0
        elif inflation_counter > 0:
            Q = Q_nominal * 10.0
            R_k = R_nominal
            inflation_counter -= 1
        else:
            Q = Q_nominal
            R_k = R_nominal

        rate = m_dot_u + (true_alpha * sigma_proc * np.random.randn())
        surge = 20.0 * np.exp(-k / 5.0) * np.random.rand() if k < 15 else 0.0
        current_weight += (rate * dt) + surge
        
        z_k = np.round((current_weight + sigma_meas * np.random.randn()) / quantization) * quantization

        F_mat = np.array([
            [1.0,  dt, m_dot_u * dt],
            [0.0, 1.0,          0.0],
            [0.0, 0.0,          1.0]
        ])
        
        x_pred = F_mat @ x_hat
        P_pred = F_mat @ P @ F_mat.T + Q
        
        y_k = z_k - (H @ x_pred)
        S_k = H @ P_pred @ H.T + R_k
        K_k = P_pred @ H.T / S_k
        
        x_hat = x_pred + K_k * y_k
        I_KH = np.eye(3) - K_k @ H
        P = I_KH @ P_pred @ I_KH.T + K_k * R_k * K_k.T
        P = (P + P.T) / 2.0

        true_x = np.array([[current_weight], [rate], [true_alpha]])
        err_x = true_x - x_hat
        
        all_nis[trial, k] = float(y_k**2 / S_k)
        all_nees[trial, k] = float(err_x.T @ np.linalg.solve(P + np.eye(3) * 1e-6, err_x))
        
        all_weight_errors[trial, k] = err_x[0, 0]
        all_flow_errors[trial, k] = err_x[1, 0]

        if trial == num_trials - 1:
            vis_data['time'].append(time_steps[k])
            vis_data['true_w'].append(current_weight)
            vis_data['kf_w'].append(x_hat[0, 0])
            vis_data['meas_w'].append(z_k)
            vis_data['true_dot_m'].append(rate)
            vis_data['kf_dot_m'].append(x_hat[1, 0])

        if is_fine and (current_weight >= (target_weight - mu_residual)):
            slow_fill_distance = current_weight - switch_weight if switch_weight is not None else 275.0
            residual_sigma = sigma_residual + max(0.0, (275.0 - slow_fill_distance) / 275.0) * 17.5
            residual_sigma = np.clip(residual_sigma, sigma_residual, 20.0)
            residual_fall = mu_residual + residual_sigma * np.random.randn()
            
            final_weights.append(current_weight + residual_fall)
            actual_end_idx = k
            
            all_nis[trial, k:] = all_nis[trial, k]
            all_nees[trial, k:] = all_nees[trial, k]
            all_weight_errors[trial, k:] = all_weight_errors[trial, k]
            all_flow_errors[trial, k:] = all_flow_errors[trial, k]
            break
            
    if len(final_weights) <= trial:
        final_weights.append(current_weight)


mean_nis = np.mean(all_nis, axis=0)
mean_nees = np.mean(all_nees, axis=0)
rmse_weight = np.sqrt(np.mean(all_weight_errors**2, axis=0))
rmse_flow = np.sqrt(np.mean(all_flow_errors**2, axis=0))


leg_font = fm.FontProperties(size=11)


fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4))
ax1 = axs2[0]
ax1.plot(vis_data['time'][:actual_end_idx], vis_data['meas_w'][:actual_end_idx], color='gainsboro', linewidth=0.8, label='Measured')
ax1.plot(vis_data['time'][:actual_end_idx], vis_data['true_w'][:actual_end_idx], 'r--', linewidth=1.2, label='True Weight')
ax1.plot(vis_data['time'][:actual_end_idx], vis_data['kf_w'][:actual_end_idx], 'b', linewidth=1.0, label='KF Estimate')
ax1.axhline(750, color='black', linestyle=':', alpha=0.5, label='Target 750g')
ax1.set_xlabel('Session Time (s)', fontweight='bold')
ax1.set_ylabel('Mass (g)', fontweight='bold')
ax1.legend(loc='upper left', prop=leg_font, framealpha=0.8)
ax1.grid(True, alpha=0.2)

ax2_left = axs2[1]
line1 = ax2_left.plot(time_steps[:actual_end_idx], rmse_weight[:actual_end_idx], 'g', linewidth=1.2, label='Mass RMSE')
ax2_left.set_xlabel('Session Time (s)', fontweight='bold')
ax2_left.set_ylabel('Mass Error Resolution (g)', fontweight='bold')
ax2_left.grid(True, alpha=0.2)

ax2_right = ax2_left.twinx()
line2 = ax2_right.plot(time_steps[:actual_end_idx], rmse_flow[:actual_end_idx], 'c', linewidth=0.9, label='Flow Rate RMSE')
ax2_right.set_ylabel('Flow Error Resolution (g/s)', fontweight='bold')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2_left.legend(lines, labels, loc='upper right', prop=leg_font, framealpha=0.8)
plt.tight_layout()
plt.savefig("kf_weight_and_rmse_isolated.png", dpi=300, bbox_inches='tight')

fig3, axs3 = plt.subplots(1, 2, figsize=(10, 4))
ax3 = axs3[0]
ax3.plot(time_steps[:actual_end_idx], mean_nis[:actual_end_idx], 'm', linewidth=1.0, label='Mean NIS')
ax3.axhline(chi2.ppf(0.95, 1), color='r', linestyle='--', linewidth=1.0, label='95% Upper Bound')
ax3.axhline(chi2.ppf(0.05, 1), color='b', linestyle='--', linewidth=1.0, label='5% Lower Bound')
ax3.set_xlabel('Session Time (s)', fontweight='bold')
ax3.set_ylabel('NIS Value', fontweight='bold')
ax3.legend(loc='upper right', prop=leg_font, framealpha=0.8)
ax3.grid(True, alpha=0.2)

ax4 = axs3[1]
ax4.plot(time_steps[:actual_end_idx], mean_nees[:actual_end_idx], 'darkblue', linewidth=1.0, label='Mean NEES')
ax4.axhline(chi2.ppf(0.95, 3), color='r', linestyle=':', linewidth=1.2, label='95% Confidence Bound')
ax4.axhline(chi2.ppf(0.05, 3), color='r', linestyle=':', linewidth=1.2)
ax4.set_yscale('log')
ax4.set_xlabel('Session Time (s)', fontweight='bold')
ax4.set_ylabel('NEES Metric (Log)', fontweight='bold')
ax4.legend(loc='upper right', prop=leg_font, framealpha=0.8)
ax4.grid(True, which="both", alpha=0.2)
plt.tight_layout()
plt.savefig("kf_consistency_tests_isolated.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\nTarget Success Rate: {sum(740.0 <= w <= 760.0 for w in final_weights) / num_trials * 100:.1f}%")
