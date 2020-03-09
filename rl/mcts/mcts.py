import numpy as np


class MCTS(object):
    """MCTS."""

    def __init__(self, game, pi, ucb, action_selection):
        """init."""
        self.game = game
        self.pi = pi
        self.ucb = ucb
        self.ac_selection = action_selection

    def reset(self):
        """Reset search tree."""
        self.Qsa = {}  # empirical q value for s, a
        self.Nsa = {}  # number of times s,a was visited
        self.Ps = {}   # initial policy for state s
        self.Ns = {}   # number of times s was visited
        self.Vs = {}   # valid moves

    def get_probs(self, s, p):
        """Get probs of state from counts."""
        ss = hash(self.game.to_string(s, p))
        counts = self.Nsa[ss]
        return counts / counts.sum()

    def search(self, s, p):
        """Perform Search."""
        s, p = self.game.get_canonical_state(s, p)
        over, score = self.game.game_over(s, p)
        if over:
            return -score

        ss = hash(self.game.to_string(s, p))
        if ss not in self.Ps:
            # leaf node
            psa, v = self.pi(s)
            self.Vs[ss] = self.game.get_valid_actions(s, p)
            psa *= self.Vs[ss]
            sum_psa = np.sum(psa)
            if sum_psa > 0.:
                self.Ps[ss] = psa / sum_psa
            else:
                raise ValueError("Zero prob of actions!!!")
            self.Ns[ss] = 0.
            self.Qsa[ss] = np.zeros(len(self.Vs[ss]))
            self.Nsa[ss] = np.zeros(len(self.Vs[ss]))
            return -v

        U = self.ucb(self.Qsa[ss], self.Ps[ss], self.Ns[ss],
                     self.Nsa[ss])
        ac = self.ac_selection(U, self.Vs[ss])

        next_s, next_p = self.game.move(s, p, ac)
        v = self.search(next_s, next_p)
        np1 = self.Nsa[ss][ac] + 1
        self.Qsa[ss][ac] = (self.Nsa[ss][ac] / np1) * self.Qsa[ss][ac] + v / np1
        self.Nsa[ss][ac] += 1
        self.Ns[ss] += 1

        return -v


if __name__ == '__main__':
    def ucb(q, p, ns, nsa):
        """Calculate upper confidence bounds."""
        return q + p * np.sqrt(ns / (nsa + 1))

    def action_selection(ucb, valid_actions):
        """Pick action with highest ucb."""
        ucb[np.logical_not(valid_actions)] = -np.inf
        return np.argmax(ucb)

    class Pi(object):
        """Random policy."""

        def __call__(self, s):
            """predict."""
            return np.ones(9) / 9, 0

    from game import TicTacToe
    game = TicTacToe()

    mcts = MCTS(game, Pi(), ucb, action_selection)

    s, p = game.reset()
    print(game.to_string(s, p))

    while not game.game_over(s, p)[0]:
        ss = hash(game.to_string(*game.get_canonical_state(s, p)))
        mcts.reset()
        for _ in range(1000):
            mcts.search(s, p)
        ac = np.argmax(mcts.Nsa[ss])
        s, p = game.move(s, p, ac)
        print(game.to_string(s, p))
    print(game.game_over(s, p))
