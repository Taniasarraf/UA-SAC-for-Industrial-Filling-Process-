import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import torch.nn as nn
import os

def plot_learning_results(history, title, output_dir, rolling_window=100):
 
    n_episodes = len(history['success'])
    window = min(rolling_window, n_episodes)
    
  
    if 'rewards' in history and len(history['rewards']) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(history['rewards'], color='darkgreen', alpha=0.3, label='Raw Reward')
        if len(history['rewards']) >= window:
            r_smoothed = np.convolve(history['rewards'], np.ones(window) / window, mode='valid')
            plt.plot(range(window-1, n_episodes), r_smoothed, color='darkgreen', linewidth=2, label=f'Rolling {window}')
        plt.title("Episodic Cumulative Reward Convergence")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "uasac_avg_reward.png"), dpi=200)
        plt.close()

    
    plt.figure(figsize=(8, 5))
    success_rate = np.convolve(history['success'], np.ones(window) / window, mode='valid')
    plt.plot(range(window-1, n_episodes), success_rate, color='blue', linewidth=2)
    plt.title(f"Operational Success Rate Tracking (Rolling {window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate Boundary")
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "uasac_success_rate.png"), dpi=200)
    plt.close()

    
    plt.figure(figsize=(8, 5))
    valid_switch_eps = [(i, s) for i, s in enumerate(history['switch_points']) if s > 0.0]
    if valid_switch_eps:
        ep_idx, sw_vals = zip(*valid_switch_eps)
        plt.scatter(ep_idx, sw_vals, s=4, alpha=0.5, color='darkorange', label='Switch Point')
        plt.title("Coarse-to-Fine Valve Cutoff Adaptation Window")
        plt.ylabel("Weight at Cutoff Point (g)")
    else:
        plt.title("Switch Point Adaptation (No Valid Sequences)")
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "uasac_switch_points.png"), dpi=200)
    plt.close()

    
    if 'mean_nis' in history and any(v > 0 for v in history['mean_nis']):
        plt.figure(figsize=(8, 5))
        plt.plot(history['mean_nis'], color='magenta', linewidth=0.8, alpha=0.6)
        plt.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5, label='Ideal Baseline (1.0)')
        plt.axhline(y=chi2_ppf_approx(0.95, 1), color='r', linestyle='--', linewidth=1.2, label='95% Chi2 Limit (3.84)')
        plt.title("Normalized Innovation Squared (NIS) Health Metric")
        plt.xlabel("Episode")
        plt.ylabel("Innovation Scale (NIS)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "uasac_nis.png"), dpi=200)
        plt.close()

    
    if 'mean_nees' in history and any(v > 0 for v in history['mean_nees']):
        plt.figure(figsize=(8, 5))
        plt.plot(history['mean_nees'], color='darkblue', linewidth=0.8, alpha=0.6)
        plt.axhline(y=3.0, color='r', linestyle='--', linewidth=1.2, label='Theoretical Expectation (3.0)')
        plt.title("Normalized Estimation Error Squared (NEES) Verification")
        plt.xlabel("Episode")
        plt.ylabel("State Covariance Error Vector (NEES)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "uasac_nees.png"), dpi=200)
        plt.close()

    
    if history['final_weights']:
        plt.figure(figsize=(8, 5))
        plt.hist(history['final_weights'], bins=50, color='skyblue', edgecolor='black', alpha=0.8)
        plt.axvline(x=750, color='r', linewidth=1.5, label='Target Line (750g)')
        plt.axvline(x=740, color='orange', linestyle='--', linewidth=1.0, label='Tolerance Gate Lower Bound')
        plt.axvline(x=760, color='orange', linestyle='--', linewidth=1.0, label='Tolerance Gate Upper Bound')
        plt.title("Completed Episode Mass Yield Boundary Map")
        plt.xlabel("Measured Yield Mass (g)")
        plt.ylabel("Frequency Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "uasac_weight_dist.png"), dpi=200)
        plt.close()

    
    if 'errors' in history and len(history['errors']) >= window:
        plt.figure(figsize=(8, 5))
        mae = np.convolve(history['errors'], np.ones(window) / window, mode='valid')
        plt.plot(range(window-1, n_episodes), mae, color='crimson', linewidth=2)
        plt.title(f"Mean Absolute Mass Tracking Deviation Profile (Rolling {window})")
        plt.xlabel("Episode")
        plt.ylabel("Mass Tracking Error Deviation (g)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "uasac_mae.png"), dpi=200)
        plt.close()

def plot_loss_convergence(actor_losses, critic_losses, output_dir):

    if len(actor_losses) == 0 or len(critic_losses) == 0:
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(critic_losses, color='darkred', alpha=0.5, linewidth=0.8, label='Critic Net Error (Loss)')
    plt.plot(actor_losses, color='darkcyan', alpha=0.5, linewidth=0.8, label='Actor Policy Variance (Loss)')
    
    
    w = min(100, len(actor_losses))
    if len(actor_losses) > w:
        c_smooth = np.convolve(critic_losses, np.ones(w)/w, mode='valid')
        a_smooth = np.convolve(actor_losses, np.ones(w)/w, mode='valid')
        plt.plot(range(w-1, len(critic_losses)), c_smooth, color='red', linewidth=1.8, label='Smoothed Critic Target')
        plt.plot(range(w-1, len(actor_losses)), a_smooth, color='teal', linewidth=1.8, label='Smoothed Actor Target')
        
    plt.title("Soft Actor-Critic Network Parameter Convergence History")
    plt.xlabel("Optimization Weight Steps")
    plt.ylabel("Calculated Objective Value Error Bounds")
    plt.yscale('symlog') 
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "uasac_loss_curves.png"), dpi=200)
    plt.close()

def save_summary_json(history, filename):
    
    n = len(history['success'])
    dt_scale = 0.01  # 10ms sampling frequency

    
    valid_indices = [i for i, s in enumerate(history['switch_points']) if s > 0.0 and i < len(history['episode_times'])]
    success_indices = [i for i in valid_indices if history['success'][i] == 1]
    
    successful_switches = [history['switch_points'][i] for i in success_indices]
    all_valid_switches = [history['switch_points'][i] for i in valid_indices]

    
    time_max_sw, time_min_sw, time_optimal_sw = 0.0, 0.0, 0.0
    if success_indices:
        sw_array = np.array(successful_switches)
        idx_max = success_indices[np.argmax(sw_array)]
        idx_min = success_indices[np.argmin(sw_array)]
        
       
        median_val = np.median(sw_array)
        idx_optimal = success_indices[np.argmin(np.abs(sw_array - median_val))]
        
        time_max_sw = float(history['episode_times'][idx_max] * dt_scale)
        time_min_sw = float(history['episode_times'][idx_min] * dt_scale)
        time_optimal_sw = float(history['episode_times'][idx_optimal] * dt_scale)

   
    conv_step = -1
    rolling_window = 100
    if n >= rolling_window:
        success_arr = np.array(history['success'])
        for idx in range(n - rolling_window + 1):
            if np.mean(success_arr[idx:idx+rolling_window]) >= 0.85:
                conv_step = int(idx + rolling_window)
                break

    summary = {
        "total_episodes": n,
        "success_rate": float(np.mean(history['success'])),
        "overflow_rate": float(np.mean(history['overflows'])),
        "underflow_rate": float(np.mean(history['underflows'])),
        "mean_absolute_error_g": float(np.mean(history['errors'])) if 'errors' in history and history['errors'] else 0.0,
        "final_weight_std_g": float(np.std(history['final_weights'])) if history['final_weights'] else 0.0,
        "convergence_episode_milestone": conv_step,
        "physical_time_diagnostics_seconds": {
            "time_at_highest_switch_point": time_max_sw,
            "time_at_lowest_switch_point": time_min_sw,
            "time_at_optimal_median_switch_point": time_optimal_sw
        },
        "switch_point_on_success": {
            "median_g": float(np.median(successful_switches)) if successful_switches else None,
            "mean_g": float(np.mean(successful_switches)) if successful_switches else None,
            "std_g": float(np.std(successful_switches)) if successful_switches else None,
            "min_g": float(np.min(successful_switches)) if successful_switches else None,
            "max_g": float(np.max(successful_switches)) if successful_switches else None,
        },
        "switch_point_all_valid": {
            "median_g": float(np.median(all_valid_switches)) if all_valid_switches else None,
            "mean_g": float(np.mean(all_valid_switches)) if all_valid_switches else None,
        },
        "kf_diagnostics": {
            "mean_nees": float(np.mean(history['mean_nees'])) if any(v > 0 for v in history['mean_nees']) else None,
            "mean_nis": float(np.mean(history['mean_nis'])) if any(v > 0 for v in history['mean_nis']) else None,
        }
    }

    with open(filename, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary data saved to JSON file: {filename}")

def chi2_ppf_approx(p, df):
    table = {
        (0.95, 1): 3.841,
        (0.05, 1): 0.004,
        (0.95, 3): 7.815,
        (0.05, 3): 0.352,
    }
    return table.get((p, df), 1.0)

def plot_sac_v_value(v_proxy, save_path, device):
    weights = np.linspace(0, 800, 100)
    test_states = [[w, 20.0, 1.0, 0.1, 0.1, 0.1, max(0, 750-w)] for w in weights]
    test_states_t = torch.FloatTensor(np.array(test_states)).to(device)
    v_values = v_proxy(test_states_t).cpu().numpy()

    plt.figure(figsize=(9, 5))
    plt.plot(weights, v_values, label="Expected Value State (V)", color='darkgreen', linewidth=2)
    plt.axvline(x=750, color='r', linestyle='--', label='Target Weight Boundary')
    plt.title("V-Value Landscape Profile View")
    plt.xlabel("Digital Twin Calculated Weight (g)")
    plt.ylabel("Value Return Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
