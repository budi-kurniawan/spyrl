import gym

from stable_baselines3 import PPO, DQN

env = gym.make('LunarLander-v2')

#model = PPO('MlpPolicy', env, verbose=1)
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
print("start evaluation")
f = open('result01.txt', 'w')

for ep in range(100):
    ep_reward = 0
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        env.render()
        
        if done:
            obs = env.reset()
    f.write(str(ep) + ',' + str(ep_reward) + '\n')
f.close()