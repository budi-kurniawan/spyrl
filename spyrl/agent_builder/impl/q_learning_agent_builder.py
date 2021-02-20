from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.q_learning_agent import QLearningAgent

class QLearningAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return QLearningAgent(self.num_actions, self.discretizer, seed, initial_policy_path)    