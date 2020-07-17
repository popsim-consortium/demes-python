import unittest

import demes
import demes.examples


class TestExamples(unittest.TestCase):
    maxDiff = None

    def dump_and_load(self, g):
        s = demes.dumps(g)
        g2 = demes.loads(s)
        s2 = demes.dumps(g2)
        g3 = demes.loads(s2)
        self.assertEqual(g2, g3)
        self.assertEqual(s, s2)

    def test_zigzag(self):
        self.dump_and_load(demes.examples.zigzag())

    def test_gutenkunst_ooa(self):
        self.dump_and_load(demes.examples.gutenkunst_ooa())

    def test_browning_america(self):
        self.dump_and_load(demes.examples.browning_america())

    def test_jacobs_papuans(self):
        self.dump_and_load(demes.examples.jacobs_papuans())

    def test_IM(self):
        self.dump_and_load(demes.examples.IM(10, 10000, 1e-5))
