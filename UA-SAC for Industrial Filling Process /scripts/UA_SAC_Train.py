import os
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

from UA_SAC_FillingEnv  import FillingEnv
from sac_utils import plot_learning_results, save_summary_json, plot_sac_v_value, plot_loss_convergence

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

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def sample(self, state):
        x = self.net(state)
        mu, log_std = self.mu(x), torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))
    def forward(self, s, a): return self.net(torch.cat([s, a], dim=1))

class VProxy(nn.Module):
    def __init__(self, q1, q2, log_alpha, obs_scale):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
        self.log_alpha = log_alpha
        self.obs_scale = obs_scale

    def forward(self, state_raw):
        state_scaled = state_raw / self.obs_scale
        device = next(self.q1.parameters()).device
        dummy_action = torch.full((state_scaled.size(0), 1), -1.0).to(device)
        with torch.no_grad():
            v = torch.min(self.q1(state_scaled, dummy_action), self.q2(state_scaled, dummy_action))
        return v

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
    with open("sac_train.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config.get('seed', 42)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    output_dir = "Outputs_Phase3_UASAC"
    os.makedirs(output_dir, exist_ok=True)

    env = FillingEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim, action_dim = 7, 1

    sac_cfg = config['sac_hyperparameters']
    max_steps = config['physics'].get('max_steps', 600)
    warmup_steps = 3000  

    obs_scale = np.array([800.0, 700.0, 2.0, 1000.0, 1000.0, 1000.0, 800.0], dtype=np.float32)

    actor = SACActor(state_dim, action_dim).to(device)
    q1, q2 = Critic(state_dim, action_dim).to(device), Critic(state_dim, action_dim).to(device)
    t_q1, t_q2 = Critic(state_dim, action_dim).to(device), Critic(state_dim, action_dim).to(device)
    t_q1.load_state_dict(q1.state_dict()); t_q2.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=sac_cfg['lr'])
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=sac_cfg['lr'])

    actor_scheduler = optim.lr_scheduler.StepLR(actor_opt, step_size=1000, gamma=0.5)
    q_scheduler = optim.lr_scheduler.StepLR(q_opt, step_size=1000, gamma=0.5)

    
    target_entropy = -2.5
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=sac_cfg['lr'])

    buffer = ReplayBuffer(100000)
    
    
    history = {k: [] for k in ['success', 'final_weights', 'switch_points', 'overflows', 
                                'underflows', 'errors', 'mean_nees', 'mean_nis', 
                                'rewards', 'episode_times']}
                                
    
    actor_loss_track = []
    critic_loss_track = []
    
    total_steps = 0
    guided = GuidedSampler(weight_lo=420, weight_hi=480)

    for ep in range(sac_cfg['episodes']):
        state, _ = env.reset()
        state_s = state / obs_scale
        done, ep_reward, step_count = False, 0, 0
        guided.reset()

        while not done:
            if total_steps < warmup_steps:
                action_np = guided.act(env.x_hat[0, 0])  
            else:
                state_t = torch.FloatTensor(state_s).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _ = actor.sample(state_t)
                action_np = action.cpu().numpy()[0]

            next_state, reward, done, _, info = env.step(action_np)
            next_state_s = next_state / obs_scale

            step_count += 1
            if step_count >= max_steps and not done:
                done = True
                reward -= 500.0
                info["status"] = "Underflow"

            buffer.push(state_s, action_np, reward, next_state_s, done)
            state_s, ep_reward = next_state_s, ep_reward + reward
            total_steps += 1

            if len(buffer) > sac_cfg['batch_size']:
                s, a, r, ns, d = [torch.FloatTensor(x).to(device)
                                   for x in buffer.sample(sac_cfg['batch_size'])]

                with torch.no_grad():
                    alpha = log_alpha.exp()
                    n_a, n_lp = actor.sample(ns)
                    target_q = torch.min(t_q1(ns, n_a), t_q2(ns, n_a)) - alpha * n_lp
                    y = r.unsqueeze(1) + (1 - d.unsqueeze(1)) * sac_cfg['gamma'] * target_q

                q_loss = F.mse_loss(q1(s, a), y) + F.mse_loss(q2(s, a), y)
                q_opt.zero_grad(); q_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 0.5)
                q_opt.step()
                critic_loss_track.append(float(q_loss.item()))

                new_a, lp = actor.sample(s)
                a_loss = (alpha * lp - torch.min(q1(s, new_a), q2(s, new_a))).mean()
                actor_opt.zero_grad(); a_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_opt.step()
                actor_loss_track.append(float(a_loss.item()))

                alpha_loss = -(log_alpha * (lp + target_entropy).detach()).mean()
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

                tau = sac_cfg['tau']
                for t, p in zip(t_q1.parameters(), q1.parameters()):
                    t.data.copy_(t.data * (1 - tau) + p.data * tau)
                for t, p in zip(t_q2.parameters(), q2.parameters()):
                    t.data.copy_(t.data * (1 - tau) + p.data * tau)

        actor_scheduler.step()
        q_scheduler.step()

        actual_switch = env.switch_weight if env.switch_weight is not None else 0.0
        history['success'].append(1 if info['status'] == "Success" else 0)
        history['final_weights'].append(info['true_weight'])
        history['switch_points'].append(actual_switch)
        history['overflows'].append(1 if info['status'] == "Overflow" else 0)
        history['underflows'].append(1 if info['status'] == "Underflow" else 0)
        history['errors'].append(abs(750.0 - info['true_weight']))
        history['mean_nees'].append(info.get('mean_nees', 0.0))
        history['mean_nis'].append(info.get('mean_nis', 0.0))
        history['rewards'].append(ep_reward)
        history['episode_times'].append(step_count)

        print(f"Episode: {ep+1:04d} | "
              f"Weight: {info['true_weight']:.1f}g | "
              f"Status: {info['status']:<9} | "
              f"Switch: {actual_switch:>5.1f}g | "
              f"Reward: {ep_reward:>7.1f} | "
              f"Steps: {step_count:>4d}")

        if (ep + 1) % 100 == 0:
            torch.save(actor.state_dict(), os.path.join(output_dir, f"uasac_actor_ep{ep+1}.pth"))
            avg_success = np.mean(history['success'][-100:])
            avg_nees = np.mean(history['mean_nees'][-100:])
            avg_nis = np.mean(history['mean_nis'][-100:])
            avg_error = np.mean(history['errors'][-100:])
            print(f"--- Ep {ep+1:04d} | Success: {avg_success:.2f} | Error: {avg_error:.2f}g | NEES: {avg_nees:.2f} | NIS: {avg_nis:.2f} ---")

    
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
    }, os.path.join(output_dir, "uasac_final_data.pth"))
    print(f"Binary comparative analytics file successfully saved.")

    torch.save(actor.state_dict(), os.path.join(output_dir, "uasac_actor_final.pth"))
    v_model = VProxy(q1, q2, log_alpha, obs_scale)
    plot_sac_v_value(v_model, os.path.join(output_dir, "v_landscape.png"), device)
    
    save_summary_json(history, os.path.join(output_dir, "uasac_stats.json"))
    plot_learning_results(history, "UA-SAC Phase 3 Analytical Profiles", output_dir)
    plot_loss_convergence(actor_loss_track, critic_loss_track, output_dir)

if __name__ == "__main__":
    train()