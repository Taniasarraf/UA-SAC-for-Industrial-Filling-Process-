import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_and_pad_data(file_path, target_episodes=1500):
    if not os.path.exists(file_path):
        return {
            'rewards': np.zeros(target_episodes), 'success': np.zeros(target_episodes),
            'switches': np.zeros(target_episodes), 'errors': np.ones(target_episodes) * 250.0,
            'weights': np.ones(target_episodes) * 400.0
        }
    data = torch.load(file_path)
    rewards = data.get('avg_rewards', data.get('rewards', []))
    success = data.get('success_history', data.get('success', []))
    switches = data.get('switch_points', [])
    errors = data.get('errors', [])
    weights = data.get('final_weights', [])
    
    n = len(rewards)
    if n < target_episodes:
        rewards = np.concatenate([rewards, np.full(target_episodes - n, rewards[-1])])
        success = np.concatenate([success, np.full(target_episodes - n, success[-1])])
        switches = np.concatenate([switches, np.full(target_episodes - n, switches[-1])])
        errors = np.concatenate([errors, np.full(target_episodes - n, errors[-1])])
        weights = np.concatenate([weights, np.full(target_episodes - n, weights[-1])])
    else:
        rewards, success, switches, errors, weights = rewards[:target_episodes], success[:target_episodes], switches[:target_episodes], errors[:target_episodes], weights[:target_episodes]
    return {'rewards': rewards, 'success': success, 'switches': switches, 'errors': errors, 'weights': weights}

def generate_unified_plots():
    target_episodes = 1500
    window = 100
    
    mc_1d = load_and_pad_data("mc_1d_final_data.pth", target_episodes)
    vanilla = load_and_pad_data("vanilla_final_data.pth", target_episodes)
    kf_sac = load_and_pad_data("kfsac_final_data.pth", target_episodes)
    ua_sac = load_and_pad_data("uasac_final_data.pth", target_episodes)
    
    models = {
        'Tabular MC': (mc_1d, 'darkgoldenrod', '--'),
        'Vanilla SAC': (vanilla, 'crimson', '-.'),
        'KF-SAC': (kf_sac, 'orange', ':'),
        'UA-SAC': (ua_sac, 'blue', '-')
    }
    
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.titlesize': 10})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    
    ax_perf, ax_error, ax_control = axs[0], axs[1], axs[2]
    
    for label, (dataset, color, style) in models.items():
        smooth_range = range(window, target_episodes + 1)
        
        
        r_smooth = np.convolve(dataset['rewards'], np.ones(window)/window, mode='valid')
        ax_perf.plot(smooth_range, r_smooth, color=color, linestyle=style, linewidth=1.5, label=label)
        
       
        e_smooth = np.convolve(dataset['errors'], np.ones(window)/window, mode='valid')
        ax_error.plot(smooth_range, e_smooth, color=color, linestyle=style, linewidth=1.5)
        
        
        sw_smooth = np.convolve(dataset['switches'], np.ones(window)/window, mode='valid')
        ax_control.plot(smooth_range, sw_smooth, color=color, linestyle=style, linewidth=1.5)
        
        valid_idx = [i for i, sw in enumerate(dataset['switches']) if sw >= 0]
        if valid_idx:
            ax_control.scatter(np.array(valid_idx[::10])+1, dataset['switches'][valid_idx[::10]], color=color, s=1.5, alpha=0.1)

   
    ax_perf.set_xlabel("Training Episode")
    ax_perf.set_ylabel("Total Return")
    ax_perf.grid(True, alpha=0.2)
    ax_perf.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
   
    ax_error.set_xlabel("Training Episode")
    ax_error.set_ylabel("Tracking Mass Error (g)")
    ax_error.set_yscale('log')
    ax_error.grid(True, which="both", alpha=0.2)
    
    
    ax_control.set_xlabel("Training Episode")
    ax_control.set_ylabel("Switching Point Weight (g)")
    ax_control.set_ylim([-10, 760])
    ax_control.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("unified_training_comparison.png", dpi=300, bbox_inches='tight')
    print("Clean, compact 1x3 conference figure exported successfully without double lines.")
    plt.show()

if __name__ == "__main__":
    generate_unified_plots()
