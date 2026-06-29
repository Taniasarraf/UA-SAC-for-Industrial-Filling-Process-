import os
import yaml
import random
import numpy as np
import torch
from collections import defaultdict
import pickle

from MonteCarlo_Baseline_FillingEnv_1V import MCFillingEnv1V
from mc_utils import save_mc_summary_json, plot_mc_learning_results

class MacroMonteCarlo1D:
    def __init__(self, cfg):
        self.epsilon = float(cfg['epsilon_start'])
        self.epsilon_min = float(cfg['epsilon_min'])
        self.epsilon_decay = float(cfg['epsilon_decay'])
        
        disc_cfg = cfg['state_discretization']
        self.w_step = int(disc_cfg['weight_step']) 
        
        
        self.low_bound = 350
        self.high_bound = 650
        self.actions = list(range(self.low_bound, self.high_bound + self.w_step, self.w_step))
        
        
        self.q_table = defaultdict(lambda: {a: float(cfg['initial_q']) for a in self.actions})
        self.visit_counts = defaultdict(lambda: {a: 0 for a in self.actions})
        self.state_key = (0,) 

    def select_action(self, evaluate=False):
        if not evaluate and (random.random() < self.epsilon):
            return random.choice(self.actions)
        
        state_actions = self.q_table[self.state_key]
        max_val = max(state_actions.values())
        best_actions = [a for a, q in state_actions.items() if q == max_val]
        return random.choice(best_actions)

    def update_policy(self, chosen_action, episode_return):
        
        self.visit_counts[self.state_key][chosen_action] += 1
        n = self.visit_counts[self.state_key][chosen_action]
        q_old = self.q_table[self.state_key][chosen_action]
        
        \
        self.q_table[self.state_key][chosen_action] = q_old + (1.0 / n) * (episode_return - q_old)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train():
    with open("MC_Train.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    seed = config.get('seed', 42)
    random.seed(seed); np.random.seed(seed)
    
    output_dir = "Outputs_MC_Baseline_1D(new)"
    os.makedirs(output_dir, exist_ok=True)
    
    env = MCFillingEnv1V()
    agent = MacroMonteCarlo1D(config['mc_hyperparameters'])
    episodes = config['mc_hyperparameters']['episodes']
    
    history = {k: [] for k in ['success', 'final_weights', 'switch_points', 'overflows', 
                                'underflows', 'errors', 'rewards', 'episode_times']}
    
    for ep in range(episodes):
        env.reset()
        
        
        chosen_threshold = agent.select_action(evaluate=False)
        action_gym = np.array([float(chosen_threshold)], dtype=np.float32)
        
       
        _, reward, done, _, info = env.step(action_gym)
        
        
        agent.update_policy(chosen_threshold, reward)
        
        actual_switch = env.switch_weight if env.switch_weight is not None else 0.0
        history['success'].append(1 if info['status'] == "Success" else 0)
        history['final_weights'].append(info['true_weight'])
        history['switch_points'].append(actual_switch)
        history['overflows'].append(1 if info['status'] == "Overflow" else 0)
        history['underflows'].append(1 if info['status'] == "Underflow" else 0)
        history['errors'].append(abs(750.0 - info['true_weight']))
        history['rewards'].append(reward)
        history['episode_times'].append(env.current_step)
        
        print(f"MC 1D Ep: {ep+1:04d} | Weight: {info['true_weight']:.1f}g | Status: {info['status']:<9} | Selected Cutoff: {chosen_threshold}g | Reward: {reward:>7.1f}")

        if (ep + 1) % 100 == 0:
            avg_success = np.mean(history['success'][-100:])
            avg_error = np.mean(history['errors'][-100:])
            best_action = max(agent.q_table[agent.state_key], key=agent.q_table[agent.state_key].get)
            print(f"--- Ep {ep+1:04d} | Success: {avg_success:.2f} | Error: {avg_error:.2f}g | Best threshold: {best_action}g | Epsilon: {agent.epsilon:.3f} ---")
            
    torch.save({
        'switch_points': np.array(history['switch_points']),
        'avg_rewards': np.array(history['rewards']),
        'errors': np.array(history['errors']),
        'final_weights': np.array(history['final_weights']),
        'success_history': np.array(history['success']),
        'episode_steps': np.array(history['episode_times'])
    }, os.path.join(output_dir, "mc_1d_final_data.pth"))
    
    save_mc_summary_json(history, os.path.join(output_dir, "mc_1d_stats.json"))
    plot_mc_learning_results(history, "MC 1-Vector Macro Threshold Baseline", output_dir)

    
   
    q_table_save_path = os.path.join(output_dir, "mc_1d_qtable.pkl")
    with open(q_table_save_path, "wb") as f:
        pickle.dump(dict(agent.q_table), f)
    print(f"\n[SUCCESS] True Trained Q-Table saved to: {q_table_save_path}")

if __name__ == "__main__":
    train()
