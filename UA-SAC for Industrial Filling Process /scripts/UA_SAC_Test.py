import torch
import numpy as np
import json
import os
from UA_SAC_FillingEnv import FillingEnv
from UA_SAC_Train import SACActor

def run_ua_sac_test(num_episodes=1000, max_steps=800):
    env = FillingEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actor = SACActor(7, 1).to(device)
    model_path = "pth"
    
    if not os.path.exists(model_path):
        print(f"ERROR: File not found at {model_path}")
        return
        
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    
    param_mean = list(actor.parameters())[0].mean().item()
    print(f"Model loaded. Sample weight mean: {param_mean:.4f}")
    
    obs_scale = np.array([800.0, 700.0, 2.0, 100.0, 300.0, 15.0, 800.0], dtype=np.float32)

    successes = 0
    weights = []
    switches = []

    print(f"Testing UA-SAC Policy...")

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0 
        
        while not done and steps < max_steps:
            state_scaled = torch.FloatTensor(state / obs_scale).unsqueeze(0).to(device)
            
            with torch.no_grad():
                x = actor.net(state_scaled)
                
                mu = actor.mu(x)
                
                if env.x_hat[0, 0] < 450.0:
                    action = np.array([1.0], dtype=np.float32)
                else:
                    action = torch.tanh(mu).cpu().numpy()[0]
                
                if ep % 200 == 0 and steps == 10:
                    print(f"DEBUG Ep {ep} | Raw Mu: {mu.item():.4f} | Final Action: {action}")
            
            state, _, done, _, info = env.step(action)
            steps += 1
        
        status = info.get('status', 'Unknown')
        sw = env.switch_weight if env.switch_weight is not None else 400.0
        
        if status == "Success": 
            successes += 1
        weights.append(info.get('true_weight', 0.0))
        switches.append(sw)
        
        if (ep + 1) % 50 == 0:
            
            print(f"\033[1mEpisode {ep+1:04d}\033[0m | \033[1mStatus:\033[0m {status:<9} | \033[1mSwitch:\033[0m {sw:>5.1f}g | \033[1mWeight:\033[0m {weights[-1]:.1f}g")

    stats = {
        "N": num_episodes,
        "Success_Rate": f"{(successes/num_episodes)*100:.2f}%",
        "MAE_g": float(np.mean(np.abs(750.0 - np.array(weights)))),
        "Mean_Switch_Point_g": float(np.mean(switches)),
        "Std_Switch_Point_g": float(np.std(switches))
    }
    
    with open("uasac_test_stats(new).json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"\n\033[1mTest Complete. Metrics saved to uasac_test_stats.json\033[0m")
    print(stats)

if __name__ == "__main__":
    run_ua_sac_test()
