import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from MC_Baseline_FillingEnv_1V import MCFillingEnv1V

def run_mc_deterministic_test(num_test_episodes=1000):
    

    env = MCFillingEnv1V()
    PKL_PATH = "Outputs_MC_Baseline_1D/mc_1d_qtable.pkl"
    
    if not os.path.exists(PKL_PATH):
        print(f"file not found at: {PKL_PATH}")
        
        return
        
    
    with open(PKL_PATH, "rb") as f:
        q_table = pickle.load(f)
    

    test_weights = []
    test_switches = []
    successes = 0
    overflows = 0
    underflows = 0

    plot_true_w = []
    plot_meas_z = []
    switch_step_idx = None

    
    state_key = (0,)

    for ep in range(num_test_episodes):
        env.reset()
        done = False
        step_idx = 0
        
        
        
        if state_key in q_table:
            state_actions = q_table[state_key]
            chosen_threshold = int(max(state_actions, key=state_actions.get))
        else:
            
            chosen_threshold = 490
            
        action_gym = np.array([float(chosen_threshold)], dtype=np.float32)
        
        
        while not done:
            
            action_idx = 1 if env.raw_weight_estimate < chosen_threshold else 0
            
            next_state, reward, done, truncated, info = env.step(action_idx)
            
            if ep == num_test_episodes - 1:
                plot_true_w.append(env.true_weight)
                plot_meas_z.append(env.raw_weight_estimate)
                if env.is_fine and switch_step_idx is None:
                    switch_step_idx = step_idx
            
            step_idx += 1
            
        final_w = info.get('true_weight', 0.0)
        switch_w = env.switch_weight if env.switch_weight is not None else 0.0
        status = info.get('status', 'Unknown')
        
        test_weights.append(final_w)
        test_switches.append(switch_w)
            
        if status == "Success":
            successes += 1
        elif status == "Overflow":
            overflows += 1
        elif status == "Underflow":
            underflows += 1
            
        if (ep + 1) % 100 == 0 or ep < 5:
            print(f"Test Ep {ep+1:04d} | True Weight: {final_w:.2f}g | Switch Point: {switch_w:.2f}g | Status: {status}")

    
    mae = float(np.mean([abs(w - 750.0) for w in test_weights]))
    mean_switch = float(np.mean([s for s in test_switches if s > 0])) if any(s > 0 for s in test_switches) else 0.0
    std_switch = float(np.std([s for s in test_switches if s > 0])) if any(s > 0 for s in test_switches) else 0.0
    success_rate = float(successes / num_test_episodes * 100)

    
    print(f"Total Evaluated Bottling Runs : {num_test_episodes}")
    print(f"Empirical Evaluation Success Rate: {success_rate:.2f}%")
    print(f"Mean Absolute Error (MAE)     : {mae:.2f}g")
    print(f"Average Policy Switch Point   : {mean_switch:.2f}g")
    print(f"Switch Point Standard Dev (σ) : {std_switch:.2f}g")
    print(f"Overflow Runs Count           : {overflows}")
    print(f"Underflow Runs Count          : {underflows}")
   

    report_data = {
        "algorithm": "Tabular Monte Carlo Baseline (Verified Fixed)",
        "total_test_episodes": num_test_episodes,
        "success_rate_percent": round(success_rate, 2),
        "mean_absolute_error_g": round(mae, 2),
        "average_switch_point_g": round(mean_switch, 2),
        "switch_point_std_g": round(std_switch, 2),
        "overflow_count": overflows,
        "underflow_count": underflows
    }
    
    with open("mc_baseline_test_report.json", "w") as f:
        json.dump(report_data, f, indent=4)

    
    plt.figure(figsize=(9, 5))
    time_axis = np.array(range(len(plot_true_w))) * 0.01
    plt.plot(time_axis, plot_meas_z, label="Quantized Measurement ($z_k$)", color='gray', alpha=0.5, drawstyle='steps-post')
    plt.plot(time_axis, plot_true_w, label="True Container Mass ($w_k$)", color='blue', linewidth=2)
    if switch_step_idx is not None:
        plt.axvline(x=switch_step_idx * 0.01, color='purple', linestyle=':', linewidth=2, label="MC Switch Trigger")
    plt.title("Deterministic Deployment Test: MC Baseline Single Episode Trajectory")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mass (grams)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mc_test_phase_trajectory.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    errors = np.array(test_weights) - 750.0
    plt.bar(range(1, num_test_episodes + 1), errors, color='indianred', edgecolor='black', alpha=0.8)
    plt.axhline(y=10.0, color='red', linestyle='--')
    plt.axhline(y=-10.0, color='red', linestyle='--')
    plt.axhline(y=0.0, color='black', linewidth=1)
    plt.title("MC Final Weight Deviations Across Unseen Evaluation Runs")
    plt.xlabel("Evaluation Run Index")
    plt.ylabel("Set-point Deviation (grams)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mc_test_phase_error_bars.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    run_mc_deterministic_test(num_test_episodes=1000)