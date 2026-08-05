import os
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from UA_TD3_FillingEnv import FillingEnv
from td3_utils import plot_learning_results, save_summary_json, plot_loss_convergence, plot_td3_q_landscape

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d
    def __len__(self): return len(self.buffer)

# Deterministic Actor for TD3
class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

# Twin Critic Network
class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 Architecture
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 Architecture
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa), self.q2_net(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa)

class GuidedSampler:
    def __init__(self, weight_lo=420, weight_hi=480):
        self.weight_lo = weight_lo
        self.weight_hi = weight_hi
        self.target_switch_weight = None

    def reset(self):
        self.target_switch_weight = np.random.uniform(self.weight_lo, self.weight_hi)

    def act(self, current_kf_weight):
        if current_kf_weight < self.target_switch_weight:
            return np.array([0.9], dtype=np.float32)
        else:
            return np.array([-0.9], dtype=np.float32)

def train():
    with open("td3_train.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config.get('seed', 42)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    output_dir = "Outputs_Phase3_UATD3(seed 42)"
    os.makedirs(output_dir, exist_ok=True)

    env = FillingEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim, action_dim = 7, 1

    td3_cfg = config['td3_hyperparameters']
    max_steps = config['physics'].get('max_steps', 600)
    warmup_episodes = config.get('warmup_steps', 50)

    obs_scale = np.array([800.0, 700.0, 2.0, 100.0, 300.0, 15.0, 800.0], dtype=np.float32)

    # Networks & Targets
    actor = TD3Actor(state_dim, action_dim, td3_cfg['hidden_dim']).to(device)
    actor_target = TD3Actor(state_dim, action_dim, td3_cfg['hidden_dim']).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = TD3Critic(state_dim, action_dim, td3_cfg['hidden_dim']).to(device)
    critic_target = TD3Critic(state_dim, action_dim, td3_cfg['hidden_dim']).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=td3_cfg['lr'])
    critic_opt = optim.Adam(critic.parameters(), lr=td3_cfg['lr'])

    actor_scheduler = optim.lr_scheduler.StepLR(actor_opt, step_size=1000, gamma=0.5)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_opt, step_size=1000, gamma=0.5)

    buffer = ReplayBuffer(100000)

    history = {k: [] for k in ['success', 'final_weights', 'switch_points', 'overflows', 
                                'underflows', 'errors', 'mean_nees', 'mean_nis', 
                                'rewards', 'episode_times']}
                                
    actor_loss_track = []
    critic_loss_track = []

    total_steps = 0
    guided = GuidedSampler(weight_lo=420, weight_hi=480)

    for ep in range(td3_cfg['episodes']):
        state, _ = env.reset()
        state_s = state / obs_scale
        done, ep_reward, step_count = False, 0, 0
        guided.reset()

        while not done:
            if ep < warmup_episodes:
                action_np = guided.act(env.x_hat[0, 0])
            else:
                state_t = torch.FloatTensor(state_s).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_t = actor(state_t)
                action_np = action_t.cpu().numpy()[0]
                
                # Add Gaussian exploration noise
                noise = np.random.normal(0, td3_cfg['exploration_noise'], size=action_dim)
                action_np = np.clip(action_np + noise, -1.0, 1.0).astype(np.float32)

            next_state, reward, done, _, info = env.step(action_np)
            next_state_s = next_state / obs_scale

            step_count += 1
            if step_count >= max_steps and not done:
                done = True
                reward -= 500.0
                info["status"] = "Underflow"

            if info["status"] != "Underflow" or random.random() < 0.1:
                buffer.push(state_s, action_np, reward, next_state_s, done)

            state_s, ep_reward = next_state_s, ep_reward + reward
            total_steps += 1

            if len(buffer) > td3_cfg['batch_size']:
                s, a, r, ns, d = [torch.FloatTensor(x).to(device) for x in buffer.sample(td3_cfg['batch_size'])]

                with torch.no_grad():
                    # Target Policy Smoothing Noise
                    noise = (torch.randn_like(a) * td3_cfg['target_policy_noise']).clamp(
                        -td3_cfg['target_noise_clip'], td3_cfg['target_noise_clip']
                    )
                    next_action = (actor_target(ns) + noise).clamp(-1.0, 1.0)

                    # Target Q Calculation
                    target_q1, target_q2 = critic_target(ns, next_action)
                    target_q = torch.min(target_q1, target_q2)
                    y = r.unsqueeze(1) + (1 - d.unsqueeze(1)) * td3_cfg['gamma'] * target_q

                # Critic Update
                current_q1, current_q2 = critic(s, a)
                critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.1)
                critic_opt.step()
                critic_loss_track.append(float(critic_loss.item()))

                # Delayed Actor & Target Network Updates
                if total_steps % td3_cfg['policy_delay'] == 0:
                    actor_loss = -critic.Q1(s, actor(s)).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.1)
                    actor_opt.step()
                    actor_loss_track.append(float(actor_loss.item()))

                    # Soft Polyak Updates
                    tau = td3_cfg['tau']
                    for t, p in zip(actor_target.parameters(), actor.parameters()):
                        t.data.copy_(t.data * (1 - tau) + p.data * tau)
                    for t, p in zip(critic_target.parameters(), critic.parameters()):
                        t.data.copy_(t.data * (1 - tau) + p.data * tau)

        actor_scheduler.step()
        critic_scheduler.step()

        actual_switch = env.switch_weight if env.switch_weight is not None else 0.0
        current_ep_nees = info.get('mean_nees', 0.0)
        current_ep_nis  = info.get('mean_nis', 0.0)

        history['success'].append(1 if info['status'] == "Success" else 0)
        history['final_weights'].append(info['true_weight'])
        history['switch_points'].append(actual_switch)
        history['overflows'].append(1 if info['status'] == "Overflow" else 0)
        history['underflows'].append(1 if info['status'] == "Underflow" else 0)
        history['errors'].append(abs(750.0 - info['true_weight']))
        history['mean_nees'].append(current_ep_nees)
        history['mean_nis'].append(current_ep_nis)
        history['rewards'].append(ep_reward)
        history['episode_times'].append(step_count)

        print(f"Episode: {ep+1:04d} | Weight: {info['true_weight']:.1f}g | Status: {info['status']:<9} | Switch: {actual_switch:>5.1f}g | Reward: {ep_reward:>7.1f} | NEES: {current_ep_nees:>6.2f} | NIS: {current_ep_nis:>5.2f} | Steps: {step_count:>4d}")

        if (ep + 1) % 100 == 0:
            torch.save(actor.state_dict(), os.path.join(output_dir, f"uatd3_actor_ep{ep+1}.pth"))
            avg_success = np.mean(history['success'][-100:])
            avg_nees = np.mean(history['mean_nees'][-100:])
            avg_nis = np.mean(history['mean_nis'][-100:])
            avg_error = np.mean(history['errors'][-100:])
            print(f"--- Ep {ep+1:04d} | Success: {avg_success:.2f} | Error: {avg_error:.2f}g | NEES: {avg_nees:.2f} | NIS: {avg_nis:.2f} ---")

    # Save Binary Comparative Analytics File
    torch.save({
        'switch_points': np.array(history['switch_points']),
        'avg_rewards': np.array(history['rewards']),
        'errors': np.array(history['errors']),
        'final_weights': np.array(history['final_weights']),
        'success_history': np.array(history['success']),
        'actor_loss': np.array(actor_loss_track),
        'critic_loss': np.array(critic_loss_track),
        'mean_nis_history': np.array(history['mean_nis']),
        'mean_nees_history': np.array(history['mean_nees']),
        'episode_steps': np.array(history['episode_times'])
    }, os.path.join(output_dir, "uatd3_final_data.pth"))
    print(f"UA-TD3 comparative analytics file successfully saved.")

    torch.save(actor.state_dict(), os.path.join(output_dir, "uatd3_actor_final.pth"))
    plot_td3_q_landscape(critic, os.path.join(output_dir, "q_landscape.png"), device, obs_scale)
    
    save_summary_json(history, os.path.join(output_dir, "uatd3_stats.json"))
    plot_learning_results(history, "UA-TD3 Phase 3 Analytical Profiles", output_dir)
    plot_loss_convergence(actor_loss_track, critic_loss_track, output_dir)

if __name__ == "__main__":
    train()