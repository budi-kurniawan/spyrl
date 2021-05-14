from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.util.util import override

class PassThroughAgentBuilder(AgentBuilder):
    def __init__(self, agent):
        super().__init__(None)
        self.agent = agent
        
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return self.agent