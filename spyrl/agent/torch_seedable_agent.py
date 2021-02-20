import torch
from spyrl.agent.seedable_agent import SeedableAgent

class TorchSeedableAgent(SeedableAgent):
    def __init__(self, seed):
        super().__init__(seed)
        if seed is not None:
            torch.manual_seed(seed)