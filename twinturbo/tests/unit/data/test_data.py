# Needed for all the corerct inputs 
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import unittest
from twinturbo.src.data.data import InMemoryDataset
import numpy as np
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".test-root")

class TestInMemoryDataset(unittest.TestCase):
    """Class for testing cs_performance"""

    def test_run(self):
        inmemorydataset = InMemoryDataset("unit/fixtures/data.h5")
        a = inmemorydataset.__getitem__(0)[0]
        np.testing.assert_allclose(a, np.array([2.36550246, 0.86485336]))

if __name__ == "__main__":
    testclass = TestInMemoryDataset()
    testclass.test_run()
