import os
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Define environment name
environment_name = 'CartPole-v0'

# Create and test the environment with random actions
env = gym.make(environment_name)
episodes = 5

print("Testing environment with random actions...\n")
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode}: Score = {total_reward}")
env.close()

# Wrap environment for Stable-Baselines3
env = DummyVecEnv([lambda: gym.make(environment_name)])

# Train a PPO model
print("\nTraining PPO model...\n")
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=20000)

# Save and reload the model
ppo_save_path = os.path.join('Training', 'Saved Models', 'PPO_model')
model.save(ppo_save_path)
del model  # Remove the model from memory
model = PPO.load(ppo_save_path, env=env)

# Evaluate the trained PPO model
print("\nEvaluating PPO model...\n")
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

# Run a single test episode with the PPO model
env = DummyVecEnv([lambda: gym.make(environment_name)])
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        print("Episode info:", info)
        break
env.close()

# Set up callbacks for stopping training on performance threshold
save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')
env = DummyVecEnv([lambda: gym.make(environment_name)])

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=stop_callback,
    eval_freq=10000,
    best_model_save_path=save_path,
    verbose=1
)

# Retrain PPO with evaluation callback
print("\nRetraining PPO model with evaluation callback...\n")
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)

# Load and evaluate the best PPO model
best_model_path = os.path.join('Training', 'Saved Models', 'best_model')
model = PPO.load(best_model_path, env=env)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

# Use a custom network architecture with PPO
print("\nTraining PPO with custom network architecture...\n")
net_arch = [dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]
model = PPO('MlpPolicy', env, verbose=1, policy_kwargs={'net_arch': net_arch})
model.learn(total_timesteps=20000, callback=eval_callback)

# Train and evaluate a DQN model
print("\nTraining and evaluating DQN model...\n")
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)

dqn_save_path = os.path.join('Training', 'Saved Models', 'DQN_model')
model.save(dqn_save_path)
model = DQN.load(dqn_save_path, env=env)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
