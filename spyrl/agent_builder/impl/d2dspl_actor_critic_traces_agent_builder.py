from spyrl.util.util import override
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.impl.d2dspl_actor_critic_traces_agent import D2DSPLActorCriticTracesAgent
from spyrl.discretiser.discretiser import Discretiser
from spyrl.normaliser.normaliser import Normaliser

class D2DSPLActorCriticTracesAgentBuilder(AgentBuilder):    
    def __init__(self, num_actions: int, discretiser: Discretiser, max_num_samples_for_classifier: int, 
                normaliser: Normaliser = None, hidden_layer_sizes = [300, 300]) -> None:
        super().__init__(num_actions, discretiser=discretiser, normaliser=normaliser)
        self.max_num_samples_for_classifier = max_num_samples_for_classifier
        self.hidden_layer_sizes = hidden_layer_sizes

    @override(AgentBuilder)
    def create_agent(self, seed, initial_policy_path):
        return D2DSPLActorCriticTracesAgent(self.num_actions, self.discretiser, self.max_num_samples_for_classifier, 
                self.normaliser, self.hidden_layer_sizes, seed)