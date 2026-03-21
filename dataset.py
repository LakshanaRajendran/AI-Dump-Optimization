import pandas as pd
import numpy as np

np.random.seed(42)

data_size = 1000

# Generate basic features
max_angle = np.random.uniform(40, 60, data_size)
hold_time = np.random.uniform(1, 8, data_size)
total_dump_time = np.random.uniform(5, 12, data_size)
vibration_rms = np.random.uniform(0.5, 2.5, data_size)
vibration_peak = np.random.uniform(1, 3, data_size)
initial_load = np.random.uniform(800, 1500, data_size)

# Create leftover logic (IMPORTANT PART)
leftover_percent = (
    30 
    - (max_angle * 0.3)
    - (hold_time * 1.5)
    - (vibration_rms * 4)
    + np.random.normal(0, 2, data_size)
)

# Clamp values between 0 and 30
leftover_percent = np.clip(leftover_percent, 0, 30)

# Create severity labels
def get_severity(x):
    if x <= 5:
        return "low"
    elif x <= 10:
        return "mild"
    elif x <= 15:
        return "moderate"
    else:
        return "severe"

severity = [get_severity(x) for x in leftover_percent]

# Create DataFrame
df = pd.DataFrame({
    "max_angle": max_angle,
    "hold_time": hold_time,
    "total_dump_time": total_dump_time,
    "vibration_rms": vibration_rms,
    "vibration_peak": vibration_peak,
    "initial_load": initial_load,
    "leftover_percent": leftover_percent,
    "severity": severity
})

# Save dataset
df.to_csv("dump_data.csv", index=False)

print("Dataset created successfully!")