import os
import json
import numpy as np
import matplotlib.pyplot as plt

def save_mc_summary_json(history, filename):
    n = len(history['success'])
    
    valid_indices = [i for i, s in enumerate(history['switch_points']) if s > 0.0]
    success_indices = [i for i in valid_indices if history['success'][i] == 1]
    
    successful_switches = [history['switch_points'][i] for i in success_indices]
    all_valid_switches = [history['switch_points'][i] for i in valid_indices]

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
        "mean_absolute_error_g": float(np.mean(history['errors'])),
        "final_weight_std_g": float(np.std(history['final_weights'])),
        "convergence_episode_milestone": conv_step,
        "switch_point_on_success": {
            "median_g": float(np.median(successful_switches)) if successful_switches else None,
            "mean_g": float(np.mean(successful_switches)) if successful_switches else None,
            "std_g": float(np.std(successful_switches)) if successful_switches else None,
        },
        "switch_point_all_valid": {
            "median_g": float(np.median(all_valid_switches)) if all_valid_switches else None,
            "mean_g": float(np.mean(all_valid_switches)) if all_valid_switches else None,
        }
    }

    with open(filename, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"MC summary analytics file successfully saved: {filename}")

def plot_mc_learning_results(history, title, output_dir, rolling_window=100):
    n_episodes = len(history['success'])
    window = min(rolling_window, n_episodes)
    
    
    plt.figure(figsize=(8, 5))
    plt.plot(history['rewards'], color='darkgoldenrod', alpha=0.3, label='Raw Reward')
    if n_episodes >= window:
        r_smoothed = np.convolve(history['rewards'], np.ones(window) / window, mode='valid')
        plt.plot(range(window-1, n_episodes), r_smoothed, color='darkgoldenrod', linewidth=2, label=f'Rolling {window}')
    plt.title(f"{title} - Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mc_avg_reward.png"), dpi=200)
    plt.close()

   
    plt.figure(figsize=(8, 5))
    success_rate = np.convolve(history['success'], np.ones(window) / window, mode='valid')
    plt.plot(range(window-1, n_episodes), success_rate, color='blue', linewidth=2)
    plt.title(f"Operational Success Rate Tracking (Rolling {window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate Percentage Bound")
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mc_success_rate.png"), dpi=200)
    plt.close()

    
    plt.figure(figsize=(8, 5))
    valid_switches = [(i, s) for i, s in enumerate(history['switch_points']) if s > 0.0]
    if valid_switches:
        ep_idx, sw_vals = zip(*valid_switches)
        plt.scatter(ep_idx, sw_vals, s=4, alpha=0.5, color='orange', label='Switch Weight')
    plt.title("Coarse-to-Fine Valve Cutoff Adaptation Window")
    plt.xlabel("Episode")
    plt.ylabel("Weight at Cutoff Point (g)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mc_switch_points.png"), dpi=200)
    plt.close()

   
    plt.figure(figsize=(8, 5))
    plt.hist(history['final_weights'], bins=50, color='gold', edgecolor='black', alpha=0.8)
    plt.axvline(x=750, color='r', linewidth=1.5, label='Target Line (750g)')
    plt.axvline(x=740, color='orange', linestyle='--', linewidth=1.0)
    plt.axvline(x=760, color='orange', linestyle='--', linewidth=1.0)
    plt.title("Completed Episode Mass Yield Boundary Map")
    plt.xlabel("Measured Yield Mass (g)")
    plt.ylabel("Frequency Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mc_weight_dist.png"), dpi=200)
    plt.close()

    
    plt.figure(figsize=(8, 5))
    mae = np.convolve(history['errors'], np.ones(window) / window, mode='valid')
    plt.plot(range(window-1, n_episodes), mae, color='crimson', linewidth=2)
    plt.title("Mean Absolute Mass Tracking Deviation Profile")
    plt.xlabel("Episode")
    plt.ylabel("Mass Tracking Error Deviation (g)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mc_mae.png"), dpi=200)
    plt.close()
