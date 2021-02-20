from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.random_agent import RandomAgent

class RandomAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed):
        return RandomAgent(self.num_actions, seed)