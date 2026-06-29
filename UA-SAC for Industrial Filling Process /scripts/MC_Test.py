import pickle
import json
import numpy as np
import os
from MonteCarlo_Baseline_FillingEnv_1V import MCFillingEnv1V

def run_mc_test(num_episodes=1000):
    env = MCFillingEnv1V()
    
    
    q_table_path = "mc_1d_qtable.pkl"
    
    if not os.path.exists(q_table_path):
        print(f"CRITICAL ERROR: Q-Table not found at {q_table_path}")
        return
        
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    
    
    best_action = max(q_table[(0,)], key=q_table[(0,)].get)
    print(f"Loaded Q-table. Optimal threshold action: {best_action}")
    
    successes = 0
    weights = []
    switches = []
    
    print(f"Testing Monte Carlo Baseline ({num_episodes} episodes)...")
    
    for ep in range(num_episodes):
        env.reset()
        _, _, _, _, info = env.step(np.array([float(best_action)]))
        
        if info.get('status') == "Success": 
            successes += 1
        weights.append(info.get('true_weight', 0.0))
        switches.append(env.switch_weight if env.switch_weight is not None else float(best_action))
        
        if (ep + 1) % 200 == 0:
            print(f"Episode {ep+1:04d} | Status: {info.get('status'):<9} | Weight: {weights[-1]:.1f}g")

    stats = {
        "N": num_episodes,
        "Success_Rate": f"{(successes/num_episodes)*100:.2f}%",
        "MAE_g": float(np.mean(np.abs(750.0 - np.array(weights)))),
        "Mean_Switch_Point_g": float(np.mean(switches)),
        "Std_Switch_Point_g": float(np.std(switches))
    }
    
    with open("mc_test_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("\nTest Complete. Metrics saved to mc_test_stats.json")
    print(stats)

if __name__ == "__main__":
    run_mc_test()
