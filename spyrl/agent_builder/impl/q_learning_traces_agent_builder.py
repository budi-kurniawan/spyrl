from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.q_learning_traces_agent import QLearningTracesAgent

class QLearningTracesAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return QLearningTracesAgent(self.num_actions, self.discretizer, seed, initial_policy_path)    