# Needed for all the corerct inputs 
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import unittest
import numpy as np
import hydra
from omegaconf import DictConfig
root = pyrootutils.setup_root(search_from=__file__, pythonpath=False, cwd=True, indicator=".test-root")

class TestInMemoryDataset(unittest.TestCase):
    """Class for testing cs_performance"""

    def test_run(self):
        hydra.initialize(version_base=None, config_path= "../fixtures")
        cfg = hydra.compose(config_name="simple_cfg.yaml")
        inmemorydataset = hydra.utils.instantiate(cfg.data)
        a = inmemorydataset.__getitem__(0)["data"]
        np.testing.assert_allclose(a, np.array([0.64768854, 1.52302986]))

if __name__ == "__main__":
    testclass = TestInMemoryDataset()
    testclass.test_run()
