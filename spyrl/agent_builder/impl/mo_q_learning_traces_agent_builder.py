from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.mo_q_learning_traces_agent import MultiObjectiveQLearningTracesAgent

class MultiObjectiveQLearningTracesAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return MultiObjectiveQLearningTracesAgent(self.num_actions, self.discretizer, self.reward_builder, seed)
