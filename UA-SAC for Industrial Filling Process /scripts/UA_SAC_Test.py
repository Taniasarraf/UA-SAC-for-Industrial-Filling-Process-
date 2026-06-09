import os
import torch
import numpy as np
from UA_SAC_FillingEnv import FillingEnv
from UA_SAC_Train import SACActor  

def run_deterministic_test(num_test_episodes=1000):
    

    
    env = FillingEnv()
    
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = SACActor(state_dim, action_dim).to(device)
    
    
    weights_path = "uasac_actor_final.pth" 
    
    if not os.path.exists(weights_path):
        print(f"file not found at: {weights_path}")
        
        return
        
    actor.load_state_dict(torch.load(weights_path, map_location=device))
    actor.eval() 
    

    
    test_weights = []
    test_switches = []
    successes = 0

    for ep in range(num_test_episodes):
        state, _ = env.reset()
        
        obs_scale = np.array([800.0, 700.0, 2.0, 1000.0, 1000.0, 1000.0, 800.0], dtype=np.float32)
        done = False
        
        while not done:
            state_s = state / obs_scale
            state_t = torch.FloatTensor(state_s).unsqueeze(0).to(device)
            
            with torch.no_grad():
                
                x = actor.net(state_t)
                mu = actor.mu(x)
                action = torch.tanh(mu).cpu().numpy()[0]
                
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            
     
        final_w = info.get('true_weight', 0.0)
        switch_w = env.switch_weight if env.switch_weight is not None else 0.0
        status = info.get('status', 'Unknown')
        
        test_weights.append(final_w)
        test_switches.append(switch_w)
            
        if status == "Success":
            successes += 1
            
        print(f"Test Ep {ep+1:02d} | True Weight: {final_w:.2f}g | Switch Point: {switch_w:.2f}g | Status: {status}")

    
   
    print(f"Total Evaluated Bottling Runs : {num_test_episodes}")
    print(f"Empirical Evaluation Success Rate: {successes / num_test_episodes * 100:.2f}%")
    print(f"Mean Absolute Error (MAE)     : {np.mean([abs(w - 750.0) for w in test_weights]):.2f}g")
    print(f"Average Policy Switch Point   : {np.mean(test_switches):.2f}g")
    print(f"Switch Point Standard Dev (σ) : {np.std(test_switches):.2f}g")
    

if __name__ == "__main__":
    run_deterministic_test(num_test_episodes=1000)