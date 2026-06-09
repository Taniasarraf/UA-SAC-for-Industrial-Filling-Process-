import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FillingEnv(gym.Env):
    def __init__(self):
        super(FillingEnv, self).__init__()
        self.target_weight = 750.0
        self.dt = 0.01
        self.quantization = 10.0
        self.coarse_rate = 600.0
        self.fine_rate = 60.0  
        self.sigma_proc = 5.0
        self.sigma_meas = 3.0
        
        self.mu_residual = 15.0
        self.sigma_residual = 2.5

        self.H = np.array([[1.0, 0.0, 0.0]])

        # Process Noise 
        self.Q_nominal = np.diag([1.0, 50.0, 1.0])   
        self.R = (self.sigma_meas**2) + ((self.quantization**2) / 12.0)
        
        self.inflation_counter = 0
        self.inflation_window = 10 
        self.current_episode = 0
        self.lambda_param = 0.02


        low = np.array([0.0, 0.0, 0.5, 0.0])
        high = np.array([800.0, 700.0, 2.0, 800.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.true_weight = 0.0
        self.current_step = 0
        self.is_fine = False
        self.prev_action = 1.0 
        self.inflation_counter = 0
        self.current_episode += 1
        self.ep_nis, self.ep_nees = [], []
        self.true_alpha = np.random.uniform(0.7, 1.5) 
        self.x_hat = np.array([[0.0], [0.0], [1.0]]) 
        self.P = np.diag([10.0, 100.0, 10.0])
        self.switch_weight = None
        return self._get_obs(), {}

    def _get_obs(self):
        
        dist_to_go = max(0.0, self.target_weight - self.x_hat[0, 0])
        return np.concatenate([self.x_hat.flatten(), [dist_to_go]]).astype(np.float32)

    def _record_switch_weight(self, w):
        if self.switch_weight is None:
            self.switch_weight = w

    def step(self, action):
        u = 1.0 if action[0] > 0.0 else -1.0

        is_first_switch  = (self.prev_action ==  1.0 and u == -1.0)
        is_revert_switch = (self.prev_action == -1.0 and u ==  1.0)

        early_switch_penalty = 0.0
        if is_first_switch and not self.is_fine:
            sw = float(self.x_hat[0, 0])
            
            
            
            if sw < 400.0:
                early_switch_penalty = -0.008 * ((400.0 - sw) ** 2)
            
            self.is_fine = True
            self.inflation_counter = self.inflation_window
            self._record_switch_weight(sw)

        chatter_penalty = 0.0
        if is_revert_switch:
            chatter_penalty = -50.0  

        self.prev_action = u

        if self.current_step < 15:
            Q = self.Q_nominal * 50.0  
        elif self.inflation_counter > 0:
            Q = self.Q_nominal * 10.0
            self.inflation_counter -= 1
        else:
            Q = self.Q_nominal

        u_for_kf = -1.0 if self.is_fine else 1.0

        
        rate = (self.fine_rate if self.is_fine else self.coarse_rate) + (self.true_alpha * self.sigma_proc * np.random.randn())
        surge = 20.0 * np.exp(-self.current_step / 5.0) * np.random.rand() if self.current_step < 15 else 0.0
        self.true_weight += (rate * self.dt) + surge
        z_k = np.round((self.true_weight + self.sigma_meas * np.random.randn()) / self.quantization) * self.quantization

        # KF State Updates
        F_mat = np.array([
            [1.0, self.dt, 0.5 * u_for_kf * (self.dt**2)],
            [0.0, 1.0,    u_for_kf * self.dt],
            [0.0, 0.0,    1.0]
        ])
        x_pred = F_mat @ self.x_hat
        P_pred = F_mat @ self.P @ F_mat.T + Q

        R_k = self.R * 10.0 if self.current_step < 15 else self.R

        S_k = self.H @ P_pred @ self.H.T + R_k
        K_k = P_pred @ self.H.T / S_k
        y_k = z_k - self.H @ x_pred
        
        self.ep_nis.append(float(y_k**2 / S_k))
        self.x_hat = x_pred + K_k * y_k
        I_KH = np.eye(3) - K_k @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K_k * R_k * K_k.T
        self.P = (self.P + self.P.T) / 2.0

        true_x = np.array([[self.true_weight], [rate], [self.true_alpha]])
        self.ep_nees.append(float((true_x - self.x_hat).T @ np.linalg.inv(self.P) @ (true_x - self.x_hat)))
        self.current_step += 1

        if not self.is_fine and self.true_weight >= (self.target_weight - self.mu_residual):
            self.true_weight += self.mu_residual
            info = {"true_weight": self.true_weight, "status": "Overflow"}
            info['mean_nis']  = float(np.mean(self.ep_nis))
            info['mean_nees'] = float(np.mean(self.ep_nees))
            return self._get_obs(), -1500.0, True, False, info

        done = self.is_fine and (self.true_weight >= (self.target_weight - self.mu_residual))
        info = {"true_weight": self.true_weight, "status": "Filling"}

        
        if self.is_fine:
            time_cost = -1.50  
        else:
            if self.x_hat[0, 0] > 450.0:       
                time_cost = -5.0  
            else:
                time_cost = -0.01  

        est_err = -self.lambda_param * min(abs(self.true_weight - self.x_hat[0, 0]), 100.0)
        reward = time_cost + est_err + chatter_penalty + early_switch_penalty

        if done:
            slow_fill_distance = self.true_weight - self.switch_weight if self.switch_weight is not None else 275.0
            residual_sigma = self.sigma_residual + max(0.0, (275.0 - slow_fill_distance) / 275.0) * 17.5
            residual_sigma = np.clip(residual_sigma, self.sigma_residual, 20.0)
            residual_fall = self.mu_residual + residual_sigma * np.random.randn()

            self.true_weight += residual_fall
            info["true_weight"] = self.true_weight
            self.x_hat[0, 0] = self.true_weight
            
            info['mean_nis']  = float(np.mean(self.ep_nis))
            info['mean_nees'] = float(np.mean(self.ep_nees))

            speed_bonus = max(0.0, (800 - self.current_step) * 0.5)

            if 740.0 <= self.true_weight <= 760.0:
                reward += 600.0 + speed_bonus
                info["status"] = "Success"
            elif self.true_weight > 760.0:
                error = self.true_weight - 760.0
                reward += -100.0 * np.log1p(error * 5.0) - 800.0 
                info["status"] = "Overflow"
            else:
                reward -= 500.0
                info["status"] = "Underflow"

        return self._get_obs(), reward, done, False, info