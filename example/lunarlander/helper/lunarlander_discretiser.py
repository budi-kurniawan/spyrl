from spyrl.discretiser.discretiser import Discretiser
from spyrl.util.util import override

class LunarLanderDiscretiser(Discretiser):
    @override(Discretiser)
    def discretise(self, state):
        pos_x, pos_y, vel_x, vel_y, angle, angular_vel, ground_contact_0, ground_contact_1 = state
        pos_x = self.do_discretise(pos_x, [-0.2, 0, 0.2])
        pos_y = self.do_discretise(pos_y, [0.1, 0.3, 0.5])
        vel_x = self.do_discretise(vel_x, [-0.25, 0, 0.25])
        vel_y = self.do_discretise(vel_y, [-0.25, 0, 0.25])
        angle = self.do_discretise(angle, [-0.25, 0, 0.25])
        angular_vel = self.do_discretise(angular_vel, [-0.25, 0, 0.25])
        ground_contact_0 = int(ground_contact_0)
        ground_contact_1 = int(ground_contact_1)
        return (pos_x * 4096 + pos_y * 1024 + vel_x * 256 + vel_y * 64 + angle * 16 + angular_vel * 4 
                + ground_contact_0 * 2 + ground_contact_1)

    def do_discretise(self, n, borders):
        count = 0
        for border in borders:
            if n < border:
                return count
            else:
                count += 1
        return count
                
    @override(Discretiser)
    def get_num_discrete_states(self):
        return 16384 # 4*4*4*4*4*4*2*2
    
    @override(Discretiser)
    def get_num_state_variables(self):
        return 8
    
class LunarLanderDiscretiser12288(LunarLanderDiscretiser):
    @override(Discretiser)
    def discretise(self, state):
        pos_x, pos_y, vel_x, vel_y, angle, angular_vel, ground_contact_0, ground_contact_1 = state
        pos_x = self.do_discretise(pos_x, [-0.2, 0.2])
        pos_y = self.do_discretise(pos_y, [0.1, 0.3, 0.5])
        vel_x = self.do_discretise(vel_x, [-0.25, 0, 0.25])
        vel_y = self.do_discretise(vel_y, [-0.25, 0, 0.25])
        angle = self.do_discretise(angle, [-0.25, 0, 0.25])
        angular_vel = self.do_discretise(angular_vel, [-0.25, 0, 0.25])
        ground_contact_0 = int(ground_contact_0)
        ground_contact_1 = int(ground_contact_1)
        return (pos_x * 3072 + pos_y * 1024 + vel_x * 256 + vel_y * 64 + angle * 16 + angular_vel * 4 
                + ground_contact_0 * 2 + ground_contact_1)

    @override(Discretiser)
    def get_num_discrete_states(self):
        return 12288 # 3*4*4*4*4*4*2*2
    
class LunarLanderDiscretiser24576(LunarLanderDiscretiser):
    @override(LunarLanderDiscretiser)
    def discretise(self, state):
        pos_x, pos_y, vel_x, vel_y, angle, angular_vel, ground_contact_0, ground_contact_1 = state
        pos_x = self.do_discretise(pos_x, [-0.5, 0.2, 0, 0.2, 0.5])
        pos_y = self.do_discretise(pos_y, [0.1, 0.3, 0.5])
        vel_x = self.do_discretise(vel_x, [-0.25, 0, 0.25])
        vel_y = self.do_discretise(vel_y, [-0.25, 0, 0.25])
        angle = self.do_discretise(angle, [-0.25, 0, 0.25])
        angular_vel = self.do_discretise(angular_vel, [-0.25, 0, 0.25])
        ground_contact_0 = int(ground_contact_0)
        ground_contact_1 = int(ground_contact_1)
        discrete_state = (pos_x * 4096 + pos_y * 1024 + vel_x * 256 + vel_y * 64 + angle * 16 + angular_vel * 4 
                + ground_contact_0 * 2 + ground_contact_1)
        if discrete_state > 24575:
            print('ds:', discrete_state)
            print(pos_x, pos_y, vel_x, vel_y, angle, angular_vel, ground_contact_0, ground_contact_1)
        return discrete_state

    @override(LunarLanderDiscretiser)
    def get_num_discrete_states(self):
        return 24576 # 6*4*4*4*4*4*2*2