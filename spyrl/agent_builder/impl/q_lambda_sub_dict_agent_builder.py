from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.impl.q_lambda_sub_dict_agent import QLambdaSubDictAgent

class QLambdaSubDictAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return QLambdaSubDictAgent(self.num_actions, self.discretiser, seed, initial_policy_path)    