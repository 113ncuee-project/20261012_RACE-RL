"""Check policy episode semantics, not PPA model correctness."""
import unittest
import numpy as np
from reverse_trial import AdaptedReinforce

class ReversePolicyTest(unittest.TestCase):
    def test_update_is_episodic_and_partial_is_flushed(self):
        agent = AdaptedReinforce(2, 1, .03, .03, 1., .005)
        x = np.array([[1., 0.], [0., 1.]])
        p = agent.probabilities(x)
        agent.update(x, 0, p, np.array([1.]), np.array([1.]), 1., False)
        np.testing.assert_allclose(agent.probabilities(x), p)
        self.assertEqual(agent.updates, 0)
        agent.finish()
        self.assertEqual(agent.updates, 1)
        self.assertGreater(agent.probabilities(x)[0], p[0])
        agent.finish()
        self.assertEqual(agent.updates, 1)
        self.assertFalse(agent.to_dict()['resumable'])

if __name__ == '__main__': unittest.main()
