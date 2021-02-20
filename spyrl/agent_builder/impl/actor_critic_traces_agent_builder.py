from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.actor_critic_traces_agent import ActorCriticTracesAgent

class ActorCriticTracesAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return ActorCriticTracesAgent(self.num_actions, self.discretiser, seed, initial_policy_path)        
    