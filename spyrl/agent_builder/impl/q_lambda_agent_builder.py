from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.impl.q_lambda_agent import QLambdaAgent

class QLambdaAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return QLambdaAgent(self.num_actions, self.discretiser, seed, initial_policy_path)    