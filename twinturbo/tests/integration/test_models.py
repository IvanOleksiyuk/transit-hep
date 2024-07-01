# import sys
# from pathlib import Path
# import os 

# current_dir = Path.cwd()
# src_path = current_dir / "src"
# sys.path.append(str(src_path))

# import unittest

# import hydra
# import numpy as np


# class TestModelConfigs(unittest.TestCase):
#     def SetUp(self):
#         # Get all the configs that should be tested
#         folder_path = "../../config/step_train_template/model"
#         files_and_dirs = os.listdir(folder_path)
#         files = [f for f in files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]
#         self.model_configs = 
#     def one_model(self, config_file):
#         hydra.core.global_hydra.GlobalHydra.instance().clear()
#         hydra.initialize(version_base=None, config_path="../fixtures")
#         cfg = hydra.compose(config_name="test_ProcessorApplyCuts.yaml")
#         dataset = hydra.utils.instantiate(cfg.data)
#         np.testing.assert_allclose(
#             dataset[0]["data"], np.array([0.7504512, 0.94056472])
#         )


# if __name__ == "__main__":
    
#     print("Success! All tests passed!")
