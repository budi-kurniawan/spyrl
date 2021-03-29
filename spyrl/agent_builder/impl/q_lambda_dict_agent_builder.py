from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.impl.q_lambda_dict_agent import QLambdaDictAgent

class QLambdaDictAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return QLambdaDictAgent(self.num_actions, self.discretiser, seed, initial_policy_path)    