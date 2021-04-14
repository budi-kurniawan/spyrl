import pickle
import os
from spyrl.util.util import override
from spyrl.agent.impl.q_lambda_dict_agent import QLambdaDictAgent
from spyrl.agent.impl.q_learning_dict_agent import QLearningDictAgent

__author__ = 'bkurniawan'

"""
A QLambdaDict agent where a policy is saved as multiple files by fragmenting the dict
"""
class QLambdaSubDictAgent(QLambdaDictAgent):

    @override(QLearningDictAgent)
    def save_policy(self, path) -> None:
        count = 0
        num_items = 500_000
        file_index = 1
        l = list(self.q.items())
        print("len(q):", len(l))
        while len(self.q) > count:
            q_temp = dict(l[count:count+num_items])
            file = open(path + '-' + str(file_index).zfill(4), 'wb')
            pickle.dump(q_temp, file)
            file.close()
            count += num_items
            file_index += 1
        # re-read the files (just for testing)
#         file_index = 1
#         q_temp = {}
#         while True:
#             p = path + '-' + str(file_index).zfill(4)
#             if not os.path.exists(p):
#                 break
#             file = open(p, 'rb')
#             temp = pickle.load(file)
#             q_temp = {**q_temp, **temp}
#             file.close()
#             file_index += 1
        
    @override(QLearningDictAgent)
    def load_policy(self, path) -> None:
        file = open(path,'rb')
        self.q = pickle.load(file)
        file.close()