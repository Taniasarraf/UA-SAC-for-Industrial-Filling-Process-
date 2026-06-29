import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MCFillingEnv1V(gym.Env):
    def __init__(self):
        super(MCFillingEnv1V, self).__init__()
       
        self.target_weight = 750.0
        self.dt = 0.01
        self.quantization = 10.0
        self.coarse_rate = 600.0
        self.fine_rate = 60.0  
        self.sigma_proc = 5.0
        self.sigma_meas = 3.0
        self.mu_residual = 15.0
        self.sigma_residual = 2.5
        
        self.action_space = spaces.Box(low=350.0, high=650.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=800.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.true_weight = 0.0
        self.current_step = 0
        self.is_fine = False
        self.prev_action = 1.0 
        self.switch_weight = None
        self.true_alpha = np.random.uniform(0.7, 1.5) 
        self.raw_weight_estimate = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        disc_z = float(np.round(self.raw_weight_estimate / self.quantization) * self.quantization)
        return np.array([disc_z], dtype=np.float32)

    def step(self, action_threshold):
        w_switch_target = float(action_threshold[0])
        total_accumulated_reward = 0.0
        done = False
        
        while not done:
            u = 1.0 if self.raw_weight_estimate < w_switch_target else -1.0
            if u == -1.0 and not self.is_fine:
                self.is_fine = True
                self.switch_weight = self.raw_weight_estimate

           
            m_dot = self.fine_rate if self.is_fine else self.coarse_rate
            surge = 20.0 * np.exp(-self.current_step / 5.0) * np.random.rand() if self.current_step < 15 else 0.0
            self.true_weight += ((m_dot + (self.true_alpha * self.sigma_proc * np.random.randn())) * self.dt) + surge
            self.raw_weight_estimate = np.round((self.true_weight + self.sigma_meas * np.random.randn()) / self.quantization) * self.quantization
            self.current_step += 1
            if not self.is_fine and self.true_weight >= (self.target_weight - self.mu_residual):
                self.true_weight += self.mu_residual
                return self._get_obs(), -1500.0, True, False, {"true_weight": self.true_weight, "status": "Overflow"}

           
            time_cost = -1.50 if self.is_fine else (-5.0 if self.raw_weight_estimate > 450.0 else -0.01)
            total_accumulated_reward += time_cost
            if self.true_weight >= (self.target_weight - self.mu_residual) or self.current_step >= 600:
                done = True

        
        res_sigma = np.clip(self.sigma_residual + max(0.0, (275.0 - (self.true_weight - (self.switch_weight or 275.0))) / 275.0) * 17.5, 2.5, 20.0)
        final_w = self.true_weight + self.mu_residual + res_sigma * np.random.randn()
        
        if 740.0 <= final_w <= 760.0:
            total_accumulated_reward += 600.0 + max(0.0, min((800 - self.current_step) * 0.5, 75.0))
            status = "Success"
        elif final_w > 760.0:
            total_accumulated_reward += -100.0 * np.log1p((final_w - 760.0) * 5.0) - 800.0
            status = "Overflow"
        else:
            total_accumulated_reward -= 500.0
            status = "Underflow"

        return self._get_obs(), total_accumulated_reward, True, False, {"true_weight": final_w, "status": status}
