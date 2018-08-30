import mdptoolbox
import mdptoolbox.example
import numpy as np


def example1():
    P, R = mdptoolbox.example.forest()
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    vi.run()
    print vi.policy     # result is (0, 0, 0)


def example2():
    P, R = mdptoolbox.example.forest()
    fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.9, 3)
    fh.run()
    print fh.V
    print fh.policy


def example3():
    P = np.array([[[0.5, 0.5], [0.8, 0.2]], [[0, 1], [0.1, 0.9]]])
    R = np.array([[5, 10], [-1, 2]])
    rvi = mdptoolbox.mdp.RelativeValueIteration(P, R)
    rvi.run()
    expected = (10.0, 3.885235246411831)
    print all(expected[k] - rvi.V[k] < 1e-12 for k in range(len(expected)))
    print rvi.average_reward
    print rvi.policy
    print rvi.iter


if __name__ == '__main__':
    print '~~~ Markov Decision Process ..'
    example3()