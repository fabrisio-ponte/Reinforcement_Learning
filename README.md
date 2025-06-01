#  RL-v3 with OpenAI Gymnasium + PPO

This project demonstrates how to simulate and train a reinforcement learning agent in the classic `CarRacing-v3` environment using `gymnasium`, `stable-baselines3`, and `PPO`.

---

##  Install Dependencies

```python
!pip install gymnasium[box2d] moviepy -q
```

## Generate Random Gameplay & Save as GIF

```python
#  Import libraries
import gymnasium as gym
import numpy as np
import imageio
import os
from IPython.display import Image

# Create CarRacing environment
env = gym.make("CarRacing-v3", render_mode="rgb_array")

# Record frames
frames = []
obs, info = env.reset()
done = False

for _ in range(300):  # ~10 seconds of driving
    frames.append(env.render())
    action = env.action_space.sample()  # Random movement
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()

# Save as GIF
os.makedirs("video", exist_ok=True)
imageio.mimsave("video/carracing.gif", frames, fps=30)

# Show GIF
Image(open("video/carracing.gif", 'rb').read())
```

# Train PPO Agent

```python
!pip install gymnasium[box2d] stable-baselines3[extra] moviepy -q
```

# PPO Training Script
```python
# 🧠 Import required modules
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import os

# 🎮 Create and wrap the environment
env_id = "CarRacing-v3"
video_folder = "./videos/"
os.makedirs(video_folder, exist_ok=True)

def make_env():
    return gym.make(env_id, render_mode="rgb_array")

env = DummyVecEnv([make_env])

# 🧠 Initialize the PPO agent
model = PPO("CnnPolicy", env, verbose=1)

# 🏋️ Train the agent
model.learn(total_timesteps=100_000)  # You can go higher, but this is a decent start
```

# Record a Video of Trained Agent
```python
# 🎥 Record a video of the trained agent
env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x == 0,
                       video_length=500, name_prefix="ppo-car")

obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)

env.close()
```

# Convert MP4 to GIF
```python
import imageio
import glob
import numpy as np
from IPython.display import Image

mp4_path = sorted(glob.glob(video_folder + "*.mp4"))[-1]
gif_path = "ppo_car.gif"

# Convert MP4 to GIF
import moviepy.editor as mpy
clip = mpy.VideoFileClip(mp4_path)
clip.write_gif(gif_path)

# 📸 Display the GIF
Image(open(gif_path,'rb').read())
```

# Folder Structure
```python
├── video/
│   └── carracing.gif
├── videos/
│   └── ppo-car-step-0.mp4
├── ppo_car.gif
├── README.md
├── write_readme.py
```

# Output
The project generates:

A short carracing.gif of random driving

A trained agent using PPO

A ppo_car.gif showcasing the trained behavior

