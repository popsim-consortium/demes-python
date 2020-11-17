import stdpopsim

import demes.convert


class TestConvertStdpopsim:
    def test_all_models_back_and_forth(self):
        # XXX: The success or failure of this test depends upon what models
        # are available in stdpopsim. This leaves open the possibility that
        # the test succeeds for one version of stdpopsim but fails for another.
        for dm1 in stdpopsim.all_demographic_models():
            g1 = demes.convert.from_stdpopsim(dm1)
            dm2 = demes.convert.to_stdpopsim(g1)
            # We don't test for equality of dm1 and dm2, because there are
            # many ways to describe the same stdpopsim/msprime model.
            # Stdpopsim does include a model equality check, but this is
            # not ideal and can require, e.g. reordering of events that occur
            # at the same time, in order for models to compare equal. We avoid
            # such awkwardness here and only check that dm1 and dm2 are
            # converted into semantically equivalent graphs.
            g2 = demes.convert.from_stdpopsim(dm2)
            g1.assert_close(g2)
