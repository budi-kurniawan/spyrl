from spyrl.discretiser.discretiser import Discretiser
from spyrl.util.util import override

FOURTHIRDS = 1.3333333333333
ONE_DEGREE = 0.0174532 # 2pi/360
SIX_DEGREES = 0.1047192
TWELVE_DEGREES = 0.2094384
FIFTY_DEGREES = 0.87266

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