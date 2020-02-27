import numpy as np


class MCTS(object):
    """MCTS."""

    def __init__(self, game, pi, ucb, action_selection):
        self.game = game
        self.pi = pi
        self.ucb = ucb
        self.ac_selection = action_selection

    def clear(self):
        self.Qsa = {}  # empirical q value for s, a
        self.Nsa = {}  # number of times s,a was visited
        self.Ps = {}   # initial policy for state s
        self.Ns = {}   # number of times s was visited
        self.Vs = {}   # valid moves

    def ucb(self, s):


    def search(self, s):
        if self.game.over(s):
            return 1

        if s not in self.Ps:
            # leaf node
            psa, v = self.pi.predict(s)
            self.Vs[s] = self.game.get_legal_actions(s, player=1)
            psa *= self.Vs[s]
            sum_psa = np.sum(psa)
            if sum_psa > 0.:
                self.Ps[s] = psa / sum_psa
            else:
                raise ValueError("Zero prob of actions!!!")
            self.Ns[s] = 0.
            self.Qsa[s] = np.zeros(len(self.Vs[s]))
            self.Nsa[s] = np.zeros(len(self.Vs[s]))
            return v

        U = self.ucb(self.Qsa[s], self.Ps[s], self.Ns[s], self.Nsa[s])
        ac = self.ac_selection(U, self.Vs[s])

        next_s, next_player = self.game.get_next_state(s, player=1, action=ac)
        next_s = self.game.get_cononical_form(next_s, next_player)
        v = self.search(next_s)
        np1 = self.Nsa[s][ac] + 1
        self.Qsa[s][ac] = (self.Nsa[s][ac] / np1) * self.Qsa[s][ac] + v / np1
        self.Nsa[s][ac] += 1
        self.Ns[s] += 1

        return -v
