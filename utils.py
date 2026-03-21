
# Efficiency Score
def efficiency_score(leftover):
    return 100 - leftover

# Simulation
def simulate(model, input_data):
    return model.predict([input_data])[0]

# Energy estimation
def estimate_energy(angle, hold, vibration):
    return (angle * 0.5) + (hold * 2) + (vibration * 3)

# Sustainability (waste saved)
def waste_saved(old, new):
    return old - new