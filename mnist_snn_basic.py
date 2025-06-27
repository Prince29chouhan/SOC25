import torch
import snntorch as snn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen, utils
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

# Training settings
batch_size = 128
data_dir = '/tmp/data/mnist'
num_steps = 10  # time steps for spiking simulation

# MNIST transform (grayscale and normalize)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# Load training data
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

# Reduce dataset size for faster experimentation
train_dataset = utils.data_subset(train_dataset, subset=10)
print("Train set size:", len(train_dataset))

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Example: simulate Bernoulli spike train for one input
raw_input = torch.ones(num_steps) * 0.5  # simulate input with 0.5 probability
spikes = torch.bernoulli(raw_input)
print("Spike vector:", spikes)
print("Spike rate: {:.2f}%".format(spikes.sum().item() * 100 / num_steps))

# Grab a batch
data_iter = iter(train_loader)
imgs, labels = next(data_iter)

# Convert images into spike trains (rate coding)
spike_data = spikegen.rate(imgs, num_steps=num_steps)
print("Spike data shape:", spike_data.shape)

# Visualize a sample (1st image in batch, 1st channel)
sample_spikes = spike_data[:, 0, 0]
print("Sample spike shape:", sample_spikes.shape)

# Animate spike frames
fig, ax = plt.subplots()
anim = splt.animator(sample_spikes, fig, ax)
HTML(anim.to_html5_video())

# Try with reduced gain
spike_data_low_gain = spikegen.rate(imgs, num_steps=num_steps, gain=0.25)
sample_spikes_low = spike_data_low_gain[:, 0, 0]

# Animate again
fig, ax = plt.subplots()
anim = splt.animator(sample_spikes_low, fig, ax)
HTML(anim.to_html5_video())

# Compare spike intensities (averaged over time)
plt.figure(figsize=(8, 4), facecolor="w")
plt.subplot(1, 2, 1)
plt.imshow(sample_spikes.mean(0).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 1')

plt.subplot(1, 2, 2)
plt.imshow(sample_spikes_low.mean(0).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 0.25')
plt.show()

# Raster plot of spikes
reshaped_spikes = sample_spikes_low.view(num_steps, -1)
plt.figure(facecolor="w", figsize=(10, 5))
splt.raster(reshaped_spikes, s=1.5, c='black')
plt.title("Input Layer Spikes")
plt.xlabel("Time Step")
plt.ylabel("Neuron Index")
plt.show()

# Look at a specific neuron (e.g. neuron 210)
neuron_idx = 210
neuron_spikes = sample_spikes.view(num_steps, -1)[:, neuron_idx]

plt.figure(figsize=(8, 1), facecolor="w")
splt.raster(neuron_spikes.unsqueeze(1), s=100, c='black', marker='|')
plt.title("Neuron 210 Spiking")
plt.xlabel("Time Step")
plt.yticks([])
plt.show()
