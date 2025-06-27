import numpy as np

# Define a basic Leaky Integrate-and-Fire (LIF) neuron
class LIF:
    def __init__(self, threshold=1.0, reset=0.0, decay=0.9, refractory=2):
        self.threshold = threshold
        self.reset_value = reset
        self.decay = decay
        self.refractory = refractory
        self.potential = 0.0
        self.last_spike_time = -1
        self.next_active_time = -1

    def step(self, input_current, t):
        if t < self.next_active_time:
            return False

        self.potential *= self.decay
        self.potential += np.sum(input_current)

        if self.potential >= self.threshold:
            self.last_spike_time = t
            self.potential = self.reset_value
            self.next_active_time = t + self.refractory
            return True
        return False

# Synaptic plasticity rule (STDP)
def apply_stdp(pre_time, post_time, w, eta, tau_plus, tau_minus):
    if pre_time > 0 and post_time > 0:
        delta = post_time - pre_time
        if delta > 0:
            w += eta * np.exp(-delta / tau_plus)
        elif delta < 0:
            w -= eta * np.exp(delta / tau_minus)
    return w

# Configuration
duration = 100
n_inputs = 5
n_hidden = 3
n_outputs = 1

# Initialize neurons
input_layer = [LIF() for _ in range(n_inputs)]
hidden_layer = [LIF() for _ in range(n_hidden)]
output_layer = [LIF() for _ in range(n_outputs)]

# Initialize random synaptic weights
w_input_hidden = np.random.rand(n_inputs, n_hidden)
w_hidden_output = np.random.rand(n_hidden, n_outputs)

# Learning parameters
eta = 0.01
tau_plus = 20
tau_minus = 20

# Define a binary spike pattern we want to detect
target_pattern = [1, 0, 1, 0, 1]

# Run simulation over defined time steps
for time in range(duration):
    # Randomly simulate input spikes (0 or 1)
    spikes_input = np.random.randint(0, 2, size=n_inputs)

    # Process input layer → hidden layer
    spike_hidden = np.zeros(n_hidden)
    for i, neuron in enumerate(input_layer):
        spike = neuron.step(spikes_input[i] * w_input_hidden[i], time)
        if spike:
            spike_hidden += w_input_hidden[i]

    # Process hidden layer → output layer
    spike_output = np.zeros(n_outputs)
    for j, neuron in enumerate(hidden_layer):
        spike = neuron.step(spike_hidden[j] * w_hidden_output[j], time)
        if spike:
            spike_output += w_hidden_output[j]

    # Final output neuron update
    for k, neuron in enumerate(output_layer):
        neuron.step(spike_output[k], time)

    # Update synaptic weights using STDP
    for i in range(n_inputs):
        for j in range(n_hidden):
            w_input_hidden[i, j] = apply_stdp(
                input_layer[i].last_spike_time,
                hidden_layer[j].last_spike_time,
                w_input_hidden[i, j],
                eta, tau_plus, tau_minus
            )
    for j in range(n_hidden):
        for k in range(n_outputs):
            w_hidden_output[j, k] = apply_stdp(
                hidden_layer[j].last_spike_time,
                output_layer[k].last_spike_time,
                w_hidden_output[j, k],
                eta, tau_plus, tau_minus
            )

    # Check if desired spike pattern occurs
    if all(neuron.last_spike_time == time for neuron, bit in zip(input_layer, target_pattern) if bit == 1):
        print(f"Pattern matched at timestep {time}")
