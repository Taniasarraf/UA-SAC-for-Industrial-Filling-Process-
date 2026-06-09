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
        self.H = np.array([[1.0, 0.0, 0.0]])
        self.current_episode = 0  

        self.action_space = spaces.Box(low=350.0, high=650.0, shape=(1,), dtype=np.float32)
        
        low = np.array([0.0], dtype=np.float32)
        high = np.array([800.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.true_weight = 0.0
        self.current_step = 0
        self.is_fine = False
        self.prev_action = 1.0 
        self.current_episode += 1  
        self.switch_weight = None
        
        self.true_alpha = np.random.uniform(0.7, 1.5) 
        self.raw_weight_estimate = 0.0
        self.raw_flow_estimate = 0.0
        
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

            is_first_switch  = (self.prev_action ==  1.0 and u == -1.0)
            is_revert_switch = (self.prev_action == -1.0 and u ==  1.0)

            early_switch_penalty = 0.0
            if is_first_switch and not self.is_fine:
                sw = float(self.raw_weight_estimate)
                if sw < 400.0:
                    early_switch_penalty = -0.008 * ((400.0 - sw) ** 2)
                
                self.is_fine = True
                self.switch_weight = sw

            chatter_penalty = -50.0 if is_revert_switch else 0.0
            self.prev_action = u

            
            current_nominal_rate = self.fine_rate if self.is_fine else self.coarse_rate
            rate = current_nominal_rate + (self.true_alpha * self.sigma_proc * np.random.randn())
            surge = 20.0 * np.exp(-self.current_step / 5.0) * np.random.rand() if self.current_step < 15 else 0.0
            
            self.true_weight += (rate * self.dt) + surge
            
            
            z_k = np.round((self.true_weight + self.sigma_meas * np.random.randn()) / self.quantization) * self.quantization

            self.raw_flow_estimate = (z_k - self.raw_weight_estimate) / self.dt
            self.raw_weight_estimate = z_k
            self.current_step += 1

            
            if not self.is_fine and self.true_weight >= (self.target_weight - self.mu_residual):
                self.true_weight += self.mu_residual
                info = {"true_weight": self.true_weight, "status": "Overflow"}
                return self._get_obs(), -1500.0, True, False, info

            done = self.is_fine and (self.true_weight >= (self.target_weight - self.mu_residual))
            info = {"true_weight": self.true_weight, "status": "Filling"}

            
            if self.is_fine:
                time_cost = -1.50  
            else:
                if self.raw_weight_estimate > 450.0:
                    time_cost = -5.0  
                else:
                    time_cost = -0.01  

            total_accumulated_reward += time_cost + chatter_penalty + early_switch_penalty

           
            if self.current_step >= 600 and not done:
                done = True
                total_accumulated_reward -= 500.0
                info["status"] = "Underflow"

        
        slow_fill_distance = self.true_weight - self.switch_weight if self.switch_weight is not None else 275.0
        residual_sigma = self.sigma_residual + max(0.0, (275.0 - slow_fill_distance) / 275.0) * 17.5
        residual_sigma = np.clip(residual_sigma, self.sigma_residual, 20.0)
        residual_fall = self.mu_residual + residual_sigma * np.random.randn()

        self.true_weight += residual_fall
        info["true_weight"] = self.true_weight
        self.raw_weight_estimate = self.true_weight
        
        speed_bonus = max(0.0, (800 - self.current_step) * 0.5)

        if 740.0 <= self.true_weight <= 760.0:
            total_accumulated_reward += 600.0 + speed_bonus
            info["status"] = "Success"
        elif self.true_weight > 760.0:
            error = self.true_weight - 760.0
            total_accumulated_reward += -100.0 * np.log1p(error * 5.0) - 800.0 
            info["status"] = "Overflow"
        else:
            total_accumulated_reward -= 500.0
            info["status"] = "Underflow"

        return self._get_obs(), total_accumulated_reward, True, False, info