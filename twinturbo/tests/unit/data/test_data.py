# Standard block for correct import
current_dir = Path.cwd()
src_path = current_dir / "src"
sys.path.append(str(src_path))

import unittest

import hydra
import numpy as np


class TestProcessors(unittest.TestCase):
    def test_ProcessorApplyCuts(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorApplyCuts.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.7504512, 0.94056472])
        )

    def test_ProcessorSplitDataFrame(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_ProcessorSplitDataFrame.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(dataset[0]["data"], np.array([0.30471708]))
        np.testing.assert_allclose(dataset[0]["aux"], np.array([-1.03998411]))


class TestDatasets(unittest.TestCase):
    """Class for testing InMemoryDataFrameDictBase subclasses."""

    def test_InMemoryDataFrameDict(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_InMemoryDataFrameDict.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.30471708, -1.03998411])
        )
        assert len(dataset) == 10
        assert dataset.get_dim()["data"] == (2,)

    def test_InMemoryDataMerge(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_InMemoryDataMerge.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.30471708, -1.03998411])
        )
        np.testing.assert_allclose(
            dataset[10]["data"], np.array([0.30471708, -1.03998411])
        )

    def test_InMemoryDataMergeClasses(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_InMemoryDataMergeClasses.yaml")
        dataset = hydra.utils.instantiate(cfg.data)
        np.testing.assert_allclose(
            dataset[0]["data"], np.array([0.30471708, -1.03998411])
        )
        np.testing.assert_allclose(
            dataset[10]["data"], np.array([0.30471708, -1.03998411])
        )
        np.testing.assert_allclose(dataset[10]["label"], np.array([1]))


class TestDatamodule(unittest.TestCase):
    def test_SimpleDataModule(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path="../fixtures")
        cfg = hydra.compose(config_name="test_SimpleDataModule.yaml")
        datamodule = hydra.utils.instantiate(cfg.data)
        assert datamodule.get_dim()["data"] == (2,)


if __name__ == "__main__":
    testclass = TestProcessors()
    testclass.test_ProcessorApplyCuts()
    testclass.test_ProcessorSplitDataFrame()
    testclass = TestDatasets()
    testclass.test_InMemoryDataFrameDict()
    testclass.test_InMemoryDataMerge()
    testclass.test_InMemoryDataMergeClasses()
    testclass = TestDatamodule()
    testclass.test_SimpleDataModule()
    print("Success! All tests passed!")