import random
from math import pi
import numpy as np
from ..fuzzy.fuzzy_controller import get_controller


class QLearning:
    def __init__(self, max_theta, max_dtheta, max_x, max_dx, n_theta, n_dtheta, n_x, n_dx, n_action, is_fuzzy=False):
        self.init_constants()

        self.max_theta = max_theta
        self.max_dtheta = max_dtheta
        self.max_x = max_x
        self.max_dx = max_dx
        self.n_action = n_action
        self.n_theta = n_theta
        self.n_dheta = n_dtheta
        self.n_x = n_x
        self.n_dx = n_dx
        self.n_action = n_action
        
        shape = ( n_x, n_dx, n_theta, n_dtheta, n_action )

        self.Q = self.initial_Q * np.ones(shape, dtype=float)
        if is_fuzzy:
            self.init_fuzzy()

    def init_constants(self):
        self.initial_Q = 0
        self.initial_fuzzy_Q = 2
        self.visits = {}

    def init_fuzzy(self):
        print("Generating fuzzy table")
        ctrl = get_controller()

        for i in range( self.Q.shape[2] ):
            if i % 10 == 0:
                print("Generating row ", i, " of ", self.Q.shape[0], " rows for Q table")
            for j in range(self.Q.shape[3]):
                _, _, theta, dtheta = self.denormalize_state( (0, 0, i, j) )
                print("\t", theta, dtheta)
                original_action = ctrl.output({'theta': theta, 'dtheta': dtheta})['force']
                normalized_action = self.normalize_action(original_action)

                for a in range(self.Q.shape[0]):
                    for b in  range(self.Q.shape[1]):
                        self.Q[(a, b,i, j, normalized_action)] = self.initial_fuzzy_Q

    def normalize_state(self, state):
        first = int( (state[2]) / ( 2.0 * self.max_theta ) * self.Q.shape[2] )
        second = int( (state[3] + self.max_dtheta) / ( 2.0 * self.max_dtheta ) * self.Q.shape[3] )
        third = int( (state[0] + self.max_x) / ( 2.0 * self.max_x ) * self.Q.shape[0] )
        fourth = int( (state[1] + self.max_dx) / ( 2.0 * self.max_dx ) * self.Q.shape[1] )

        return ( third, fourth, first, second  )

    def denormalize_state(self, state):
        first = (float(state[2]) / self.Q.shape[2] * 2.0 * self.max_theta)
        second = (float(state[3]) / self.Q.shape[3] * 2.0 * self.max_dtheta - self.max_dtheta)
        third = (float(state[0]) / self.Q.shape[0] * 2.0 * self.max_x - self.max_x)
        fourth = (float(state[1]) / self.Q.shape[1] * 2.0 * self.max_dx - self.max_dx)

        return ( third, fourth, first, second )

    def denormalize_action(self, action):
        half = (self.n_action - 1) / 2
        if action == half:
            return 0
        else:
            return 2 / (self.n_action - 1) * (action - half)

    def sgn(self, x):
        if x >= 0:
            return 1
        else: 
            return -1

    def normalize_action(self, action):
        if abs(action) < 0.0001:
            return int( (self.n_action - 1) / 2 )
        else:
            return int( (self.n_action - 1) / 2 + action / (2 / (self.n_action - 1) ) )

    def action(self, state, k = 3):
        state = self.normalize_state(state)

        actions = self.Q[state]

        normalization_factor = None
        minimal_action = min(actions)
        if minimal_action < 0:
            normalization_factor = -minimal_action
        else:
            normalization_factor = 0

        actions = [ (i, actions[i]) for i in range(len(actions)) ]
        max_action = max(actions, key=lambda x: x[1])
        return max_action[0], self.denormalize_action(max_action[0])

        probabilities = []
        total = 0
        for a in range( len(actions) ):
            curr_probability = k ** ( self.Q[ tuple( list(state) + [a] ) ] + normalization_factor )
            probabilities.append(total + curr_probability)
            total = total + curr_probability

        probabilities = [p / total for p in probabilities]

        chance = random.random()
        for i in range(len(probabilities)):
            if chance < probabilities[i]:
                return i, self.denormalize_action(i)

    def update(self, s, a, next_s, r, gamma = 0.7, alpha=1):

        if tuple(list(s) + [a]) not in self.visits:
            self.visits[ tuple(list(s) + [a]) ] = 1
        else:
            self.visits[ tuple(list(s) + [a]) ] += 1
        
        alpha = 1
        
        s = self.normalize_state(s)
        next_s = self.normalize_state(next_s)

        max_action = max( list(self.Q[ tuple(next_s) ]) )
        self.Q[ tuple( list(s) + [a] ) ] = self.Q[ tuple( list(s) + [a] ) ] + \
            alpha * ( r + gamma * max_action - self.Q[ tuple(list(s) + [a]) ])

if __name__ == "__main__":
    q = QLearning(89*pi / 180, 100, 100, 100, 200)

    q_str = ""
    for i in range( q.Q.shape[0] ):
        for j in range(q.Q.shape[0]):
            for k in range(q.Q.shape[0]):
                q_str +=  str( q.Q[ (i, j, k) ] ) + " "
            q_str += "\n"
        q_str += "\n"

    print(q_str)
