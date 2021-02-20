from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.mo_actor_critic_traces_agent import MultiObjectiveActorCriticTracesAgent

class MultiObjectiveActorCriticTracesAgentBuilder(AgentBuilder):
    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return MultiObjectiveActorCriticTracesAgent(self.num_actions, self.discretizer, self.reward_builder, seed)
