# SpyRL

SpyRL is a reinforcement learning (RL) framework that provides commonly used library, so you will write
less code. For example, it automatically creates a learning graph for every learning session.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install SpyRL.

```bash
pip install spyrl
```

## Usage

```python
import gym
import sys
sys.path.insert(0, "../spyrl")
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.agent_builder.impl.actor_critic_traces_agent_builder import ActorCriticTracesAgentBuilder
from example.cartpole.helper.cartpole_discretiser import CartpoleDiscretiser
from example.cartpole.helper.env_wrapper import GymEnvWrapper

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    num_actions = env.action_space.n
    config = ActivityConfig(num_episodes=1000, out_path='result/cartpole/test1/')
    agent_builder = ActorCriticTracesAgentBuilder(num_actions, discretiser=CartpoleDiscretiser())
    learning = Learning(listener=BasicFunctions(render=False))
    learning.learn(env, agent_builder, config)```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Citation

To cite this repository in publications:
@misc{spyrl,
  author = {Kurniawan, Budi and Vamplew, Peter and Papasimeon, Michael and Dazeley, Richard and Foale, Cameron},
  title = {SpyRL Framework},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/budi-kurniawan/spyrl}},
}

## License
...
