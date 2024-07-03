import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import os 
import subprocess
import unittest


class TestModelConfigs(unittest.TestCase):
    def SetUp(self):
        # Get all the configs that should be tested
        folder_path = str(root) + "/twinturbo/config/step_train_template/model"
        files_and_dirs = os.listdir(folder_path)
        self.model_configs = [f for f in files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]
        self.model_configs.sort()
        print(self.model_configs)
    def one_model(self, config_file):
        command = "/opt/conda/bin/python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py"\
                  " --config-name twinturbo_test_model"\
                 f" step_train_template/model={config_file}"\
                  " data=gauss_corr_2_10K_twinturbo_usem"
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(config_file)
        print()
        print(command)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        result = subprocess.run(command, shell=True, check=True) #stdout=subprocess.DEVNULL
        
    def test_all_configs(self):
        for config in self.model_configs[:1]:
            self.one_model(config)


if __name__ == "__main__":
    test_model_configs = TestModelConfigs()
    test_model_configs.SetUp()
    file_names = [
        # 'twinturbo_reco-aclip.yaml',
        # 'twinturbo_reco-acontr.yaml',
        # 'twinturbo_reco-cons-acontr-vic.yaml',
        # 'twinturbo_reco-cons-acontr.yaml',
        # 'twinturbo_reco-cons-acontr0.001both.yaml',
        # 'twinturbo_reco-cons-acontr0.005a0.05.yaml',
        # 'twinturbo_reco-cons-acontr0.01a0.02.yaml',
        # 'twinturbo_reco-cons-acontr0.01a0.05.yaml',
        # 'twinturbo_reco-cons-acontr0.02a0.01.yaml',
        # 'twinturbo_reco-cons-acontr0.05a0.01.yaml',
        # 'twinturbo_reco-cons-acontr0.1both.yaml',
        # 'twinturbo_reco-cons-aclip.yaml',
        # 'twinturbo_reco-cons.yaml',
        # 'twinturbo_reco-cons-acontr0.01a0.05.yaml',
        # 'twinturbo_reco0.1-cons-acontr.yaml',
        # 'twinturbo_reco_DisCo.yaml',
        #'twinturbo_reco_DisCo_nonormvar.yaml',
        #'twinturbo_reco_DisCo_wide.yaml',
        #'twinturbo_reco_cons_DisCo.yaml',
        'twinturbo_reco_cons_DisCo_wide.yaml',
        'twinturbo_reco_cons_DisCo_wide_cons1.yaml',
        'twinturbo_reco_cons_DisCo_wide_cons1_adam.yaml',
        'twinturbo_reco_adam.yaml',
        'twinturbo_reco_noise_mINx.yaml',
        'twinturbo_reco_noise_mSEP.yaml',
        'twinturbo_reco_only.yaml',
        'twinturbo_reco_only_nom.yaml',
        'twinturbo_reco_pearson.yaml',
        'twinturboKAN_reco_DisCo.yaml'
    ]
    for file_name in file_names:
        test_model_configs.one_model(file_name)

    print("Success! All tests passed!")

