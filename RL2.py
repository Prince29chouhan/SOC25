import os
import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Set up environment and basic paths
env_id = 'LunarLander-v2'
env = DummyVecEnv([lambda: gym.make(env_id)])

models_dir = os.path.join('models', 'LunarLander')
logs_dir = os.path.join('logs', 'LunarLander')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Set up evaluation callback to stop training once reward threshold is hit
reward_limit = 200
stop_train = StopTrainingOnRewardThreshold(reward_threshold=reward_limit, verbose=1)
eval_cb = EvalCallback(env,
                       callback_on_new_best=stop_train,
                       eval_freq=10000,
                       best_model_save_path=models_dir,
                       verbose=1)


print("Training PPO model on LunarLander...")
ppo_model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)
ppo_model.learn(total_timesteps=50000, callback=eval_cb)

# Save PPO model
ppo_model_path = os.path.join(models_dir, 'ppo_lander')
ppo_model.save(ppo_model_path)

# Evaluate PPO model
print("Evaluating PPO model...")
evaluate_policy(ppo_model, env, n_eval_episodes=10, render=True)


print("Training A2C model with a custom network...")

custom_layers = [dict(pi=[256, 256], vf=[256, 256])]

a2c_model = A2C('MlpPolicy',
                env,
                policy_kwargs={'net_arch': custom_layers},
                verbose=1,
                tensorboard_log=logs_dir)
a2c_model.learn(total_timesteps=50000, callback=eval_cb)

# Save and reload A2C model
a2c_model_path = os.path.join(models_dir, 'a2c_lander')
a2c_model.save(a2c_model_path)
a2c_model = A2C.load(a2c_model_path, env=env)

# Final evaluation
print("Evaluating A2C model...")
evaluate_policy(a2c_model, env, n_eval_episodes=10, render=True)

env.close()
