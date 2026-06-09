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
        self.fine_rate = 20.0
        self.sigma_proc = 5.0
        self.sigma_meas = 3.0
        self.true_alpha = 1.2
        
        self.H = np.array([[1.0, 0.0, 0.0]])
        
        self.Q_nominal = np.diag([0.01, 1.0,0.2]) 
        self.R = (self.quantization**2) / 12.0 
        
        # Observation: [w_hat, rate_hat, alpha_hat, P11, P22, P33, dist_to_target]
        low = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        high = np.array([850.0, 700.0, 2.0, 1000.0, 1000.0, 1000.0, 850.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.true_weight = 0.0
        self.current_step = 0
        self.is_fine = False
        self.switched_this_step = False
        
    
        self.x_hat = np.array([[0.0], [0.0], [1.0]]) 
        self.P = np.diag([1.0, 1.0, 100.0]) 
        return self._get_obs(), {}

    def _get_obs(self):
        p_diag = np.diag(self.P)
        dist_to_go = max(0.0, self.target_weight - self.x_hat[0,0])
        return np.concatenate([
            self.x_hat.flatten(), 
            p_diag, 
            [dist_to_go]
        ]).astype(np.float32)

    def step(self, action):
        
        u = 1.0 if action[0] > 0.0 else -1.0
        
        
        if u == -1.0 and not self.is_fine:
            self.is_fine = True
            self.switched_this_step = True
            Q = self.Q_nominal * 10.0 
        else:
            Q = self.Q_nominal
            self.switched_this_step = False

        
        rate = (self.fine_rate if self.is_fine else self.coarse_rate) + \
               (self.true_alpha * self.sigma_proc * np.random.randn())
        
        
        surge = 20.0 * np.exp(-self.current_step / 5.0) * np.random.rand() if self.current_step < 15 else 0.0
        self.true_weight += (rate * self.dt) + surge
        
        
        z_k = np.round((self.true_weight + self.sigma_meas * np.random.randn()) * self.quantization) / self.quantization
        
        
        F = np.array([[1.0, self.dt, 0.5 * u * (self.dt**2)],
                      [0.0, 1.0, u * self.dt],
                      [0.0, 0.0, 1.0]])
        
        x_pred = F @ self.x_hat
        P_pred = F @ self.P @ F.T + Q
        
        y_k = z_k - (self.H @ x_pred)
        S_k = self.H @ P_pred @ self.H.T + self.R
        K_k = P_pred @ self.H.T / S_k
        
        self.x_hat = x_pred + K_k * y_k
        
        
        self.x_hat[0,0] = max(0.001, self.x_hat[0,0])
        self.x_hat[2,0] = max(0.001, self.x_hat[2,0])
        
        # Joseph Form 
        I_KH = np.eye(3) - K_k @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K_k * self.R * K_k.T
        self.P = (self.P + self.P.T) / 2.0 # Force Symmetry
        
        self.current_step += 1
        done = self.true_weight >= self.target_weight
        
        
        reward = -0.1 
        
       
        est_error = np.abs(self.true_weight - self.x_hat[0,0])
        reward -= 0.5 * est_error 
        
       
        info = {"true_weight": self.true_weight}
        if done:
            final_error = np.abs(self.true_weight - self.target_weight)
            
            reward -= (final_error * 10.0) 
            
            if 740.0 <= self.true_weight <= 760.0:
                reward += 500.0
                info["status"] = "Success"
            else:
                reward -= 500.0 
                info["status"] = "Failure"
        else:
            info["status"] = "Filling"
            
        return self._get_obs(), reward, done, False, info