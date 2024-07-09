import copy
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pickle
from twinturbo.src.data.lhco_curtains import convert_lhco_to_curtain_format
from twinturbo.src.data.cathode_preprocessing import CathodePreprocess
import torch

OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "in": np.isin,
    "notin": lambda x, y: ~np.isin(x, y),
}

##############################################
# Here data is always considered to be a list of pandas dataframes other

##############################################
# Collection of functions for data pre-processing
##############################################


class ProcessorIntervals:
    def __init__(self, scalar_df_name, var_name, intervals):
        self.scalar_df_name = scalar_df_name
        self.var_name = var_name
        self.intervals = intervals

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # find the events that pass the cuts
        for interval in self.intervals:
            min_val, max_val = interval[0], interval[1]
            bool_indices_new = data[self.scalar_df_name][self.var_name] < max_val
            bool_indices_new &= data[self.scalar_df_name][self.var_name] >= min_val
            if interval == self.intervals[0]:
                bool_indices = bool_indices_new
            else:  # combine the intervals
                bool_indices |= bool_indices_new
        # apply the cuts to all the dataframes
        for key, value in data.items():
            data[key] = value[bool_indices]
        return data

class ProcessorApplyCuts:
    def __init__(self, scalar_df_name, cuts):
        self.scalar_df_name = scalar_df_name
        self.cuts = cuts

    @staticmethod
    def process_cut_string(string):
        var_name, operator_s, value_s = string.split(" ")
        return var_name, OPERATORS[operator_s], float(value_s)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # find the events that pass the cuts
        for cut in self.cuts:
            var_name, operator, value = self.process_cut_string(cut)
            indices = operator(data[self.scalar_df_name][var_name], value)
        # apply the cuts to all the dataframes
        for key, value in data.items():
            data[key] = value[indices]
        return data
  
class ProcessorSplitDataFrameVars:
    def __init__(self, frame_name, new_df_dict):
        self.frame_name = frame_name
        self.new_df_list = new_df_dict

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_new = {}
        for key, value in data.items():
            if key == self.frame_name:
                for new_key, var_list in self.new_df_list.items():
                    data_new[new_key] = value[var_list]
            else:
                data_new[key] = value
        return data_new


class ProcessorToFloat32:
    def __init__(self, frame_names):
        self.frame_names = frame_names

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for name in self.frame_names:
            data[name] = data[name].astype(np.float32)
        return data


class ProcessorShuffle:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # find the events that pass the cuts
        for key, value in data.items():
            data[key] = value.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
        return data


class ProcessorNormalize():
    def __init__(self, frame_names = None, load_normaliser_file= None, save_normaliser_file=None):
        self.frame_names = frame_names
        self.load_normaliser_file = load_normaliser_file
        self.save_normaliser_file = save_normaliser_file

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.load_normaliser_file is not None:
            with open(self.load_normaliser_file, "rb") as f:
                normaliser = pickle.load(f)
            use_loaded = True
        else:
            normaliser = {}
            use_loaded = False
        frame_names = self.frame_names
        if frame_names is None:
            if use_loaded: # if no frame names are provided but normaliser is loaded, use the loaded frame names
                frame_names = normaliser.keys()
            else: # if no frame names are provided and no normaliser is loaded, use all the frames
                frame_names = data.keys()
        for name in frame_names:
            if use_loaded:
                mean = normaliser[name]["mean"]
                std = normaliser[name]["std"]
            else:
                mean = data[name].mean()
                std = data[name].std()
            #Scale the data accordingly 
            data[name] = (data[name] - mean) / std
            normaliser.update({name: {"mean": mean, "std": std}})
        if self.save_normaliser_file is not None:
            with open(self.save_normaliser_file, "wb") as f:
                pickle.dump(normaliser, f)
        return data

class ProcessorLHCOcurtains():
    def __init__(self, frame_name):
        self.frame_name = frame_name
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.frame_name] = convert_lhco_to_curtain_format(data[self.frame_name])
        return data
    
class ProcessorSignalContamination():
    def __init__(self, frame_name, var_name=None, n_contamination=0, invert=False, no_background=False):
        self.frame_name = frame_name
        self.invert = invert
        if var_name is None:
            self.var_name = frame_name
        else:
            self.var_name = var_name
        self.n_contamination = n_contamination
        self.no_background = no_background

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        indices_bkg = data[self.frame_name].index[data[self.frame_name][self.var_name]==False].tolist()
        indices_sig = data[self.frame_name].index[data[self.frame_name][self.var_name]==True].tolist()
        if self.no_background:
            if self.invert:
                indices_all = indices_bkg[:self.n_contamination]
            else:
                indices_all = indices_sig[:self.n_contamination]
        else:
            if self.invert:
                indices_all = indices_sig + indices_bkg[:self.n_contamination]
            else:
                indices_all = indices_bkg + indices_sig[:self.n_contamination]
        # apply the cuts to all the dataframes
        for key, value in data.items():
            data[key] = value.loc[indices_all]
        return data

class ProcessorRemoveFrames():
    def __init__(self, frame_names):
        self.frame_names = frame_names

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for name in self.frame_names:
            data.pop(name)
        return data

class ProcessorCATHODE():
    def __init__(self, frame_name, save_pkl=None, load_pkl=None):
        self.frame_name = frame_name
        self.load_pkl = load_pkl
        self.save_pkl = save_pkl
        

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:

        np_data = data[self.frame_name].to_numpy()
        if self.load_pkl is not None:
            with open(self.load_pkl, "rb") as f:
                cathode_preprocessor = pickle.load(f)
            print("preprocessor is loadded from a file")
        else:
            cathode_preprocessor = CathodePreprocess()
            cathode_preprocessor.fit(torch.Tensor(np_data))
        
        column_names = data[self.frame_name].columns
        np_processed = cathode_preprocessor.transform(torch.Tensor(np_data)).numpy()
        data[self.frame_name] = pd.DataFrame(np_processed, columns=column_names)
        
        if self.save_pkl is not None:
            with open(self.save_pkl, "wb") as f:
                pickle.dump(cathode_preprocessor, f)
        return data

class ProcessorMergeFrames():
    def __init__(self, frame_names, new_frame_name):
        self.frame_names = frame_names
        self.new_frame_name = new_frame_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.new_frame_name] = pd.concat([data[name] for name in self.frame_names], axis=1)
        return data
##############################################

class InMemoryDataFrameDictBase(Dataset):
    """Basic class for in-memory datasets.
    data is stored as a dictionary of pandas DataFrames.
    """

    def __getitem__(self, item):
        if self.list_order is None:
            return {
                key: value.iloc[item].to_numpy() for key, value in self.data.items()
            }
        return [self.data[it].to_numpy()[item] for it in self.list_order]

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def get_dim(self):
        if self.list_order is None:
            return {
                key: value.iloc[0].to_numpy(dtype=np.float32).shape
                for key, value in self.data.items()
            }
        return [
            self.data[it].to_numpy(dtype=np.float32)[0].shape for it in self.list_order
        ]

    def load(self, file_path: str) -> pd.DataFrame:
        data = {}

        # Open the HDF5 file
        with pd.HDFStore(file_path, "r") as store:
            # Iterate over all the keys (dataset names) in the file
            for key in store:
                # Read each dataset into a pandas DataFrame and store in the dictionary
                data[key[1:]] = store[key]
        return data

    def shuffle(self, random_state=42):
        for key, value in self.data.items():
            self.data[key] = value.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
        print("Data shuffled, please mind all later combinations!")

    def split(self, n_train):
        """Very inefficient way to split data."""
        train_data = {key: value.iloc[:n_train] for key, value in self.data.items()}
        val_data = {key: value.iloc[n_train:] for key, value in self.data.items()}
        self.data = train_data
        rest = copy.deepcopy(self)
        rest.data = val_data
        return self, rest

    def plot(self, plot_dir):
        print("Plotting the data")
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        for key, value in self.data.items():
            plt.figure()
            sns.pairplot(value)
            plt.savefig(Path(plot_dir) / f"{key}.png")
            plt.close()

    def init_processors(self, processor_cfg):
        self.processors = []
        for processor in processor_cfg:
            self.processors.append(processor)

    def apply_processors(self, data):
        for processor in self.processors:
            data = processor(data)
        return data

    # Functions that you casn call if you really need them not in configs
    def merge_dataframes(self, frame_names, new_frame_name):
        self.data[new_frame_name] = pd.concat([self.data[name] for name in frame_names], axis=1)
        self.list_order.append(new_frame_name)

    def write_npy(self, file_path: str, keys=None, save_file_names=None):
        if save_file_names is not None:
            assert len(save_file_names) == len(keys)
        if save_file_names is None:
            if keys is None:
                keys = self.data.keys()
            i=0
            for key in keys:
                np.save(file_path + save_file_names[i] + ".npy", self.data[key].to_numpy())
                i+=1
        else:
            if keys is None:
                keys = self.data.keys()
            for key in keys:
                np.save(file_path + key + ".npy", self.data[key].to_numpy())

class InMemoryDataFrameDict(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames.
    Loaded from an HDF5 file. Preprocessing is applied.
    """

    def __init__(self, file_path: str, processor_cfg=[], list_order=None, plotting_path= None) -> None:
        self.file_path = file_path
        self.list_order = list_order
        self.data = self.load(file_path)
        
        self.init_processors(processor_cfg)
        self.data = self.apply_processors(self.data)
        
        if plotting_path is not None:
            self.plot(plotting_path)

class InMemoryDataMerge(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames.
    Merged from 2 or more InMemoryDataFrameDict objects.
    """

    def __init__(self, dataset_list, do_shuffle=False) -> None:
        self.list_order = dataset_list[0].list_order
        if isinstance(self.list_order, list):
            self.list_order.append("label")
        self.data = {}
        for i, dataset in enumerate(dataset_list):
            if i == 0:
                for key, value in dataset.data.items():
                    self.data[key] = value
            else:
                for key, value in dataset.data.items():
                    self.data[key] = pd.concat([self.data[key], value])
        if do_shuffle:
            self.shuffle()

class InMemoryDataMergeClasses(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames.
    Merged from 2 or more InMemoryDataFrameDict or simmilar objects.
    The lable df is added to them with class_lables.
    """

    def __init__(
        self, dataset_list, class_lables=None, do_shuffle=True, sample_to_min=True, plotting_path= None, processor_cfg=[], 
    ) -> None:
        self.list_order = dataset_list[0].list_order
        if isinstance(self.list_order, list):
            self.list_order.append("label")
        if class_lables is None:
            class_lables = list(range(len(dataset_list)))
        self.data = {}
        if sample_to_min:
            min_len = min([len(dataset) for dataset in dataset_list])
            for i, dataset in enumerate(dataset_list):
                if i == 0:
                    for key, value in dataset.data.items():
                        self.data[key] = value.iloc[:min_len]
                    self.data["label"] = pd.DataFrame(
                        np.array([class_lables[i]] * min_len).reshape(-1, 1),
                        columns=["label"],
                    )
                else:
                    for key, value in dataset.data.items():
                        self.data[key] = pd.concat([
                            self.data[key],
                            value.iloc[:min_len],
                        ])
                    self.data["label"] = pd.concat([
                        self.data["label"],
                        pd.DataFrame(
                            np.array([class_lables[i]] * min_len).reshape(-1, 1),
                            columns=["label"],
                        ),
                    ])
        else:
            for i, dataset in enumerate(dataset_list):
                if i == 0:
                    for key, value in dataset.data.items():
                        self.data[key] = value
                    self.data["label"] = pd.DataFrame(
                        np.array([class_lables[i]] * len(value)).reshape(-1, 1),
                        columns=["label"],
                    )
                else:
                    for key, value in dataset.data.items():
                        self.data[key] = pd.concat([self.data[key], value])
                    self.data["label"] = pd.concat([
                        self.data["label"],
                        pd.DataFrame(
                            np.array([class_lables[i]] * len(value)).reshape(-1, 1),
                            columns=["label"],
                        ),
                    ])
        if do_shuffle:
            self.shuffle()

        self.init_processors(processor_cfg)
        self.data = self.apply_processors(self.data)

        if plotting_path is not None:
            self.plot(plotting_path)
    
class SimpleDataModule(LightningDataModule):
    def __init__(
        self,
        loader_kwargs,
        train_data = None,
        val_data = None,
        train_frac: float = 0.8,
        test_data=None,
    ) -> None:
        super().__init__()
        self.loader_kwargs = loader_kwargs
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        if train_data is None:
            print("No training dataset config")
        else:
            print("Training dataset loaded using cofig")

        if val_data is not None:
            print("Validation dataset loaded using cofig")
        elif self.train_data is not None:
            if train_frac is not None:
                n_data = len(self.train_data)
                n_train = int(train_frac * n_data)
                self.train_data, self.val_data = self.train_data.split(n_train)
                print(
                    "Validation dataset is is split "
                    f"from training with fraction {1 - train_frac}"
                )
            else:
                self.val_data = None
                print("No validation dataset set")
        else:
            self.val_data = None
            print("No validation dataset set")

        if test_data is None:
            print("No test dataset set")
        else:
            print("Test dataset set from config")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, **self.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, **self.loader_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, **self.loader_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dim(self):
        return self.train_data.get_dim()

    def get_var_group_list(self):
        var_group_list = []
        for key in self.train_data.list_order:
            var_group_list.append(self.train_data.data[key].columns.tolist())
        return var_group_list

class CombDataset(InMemoryDataFrameDictBase):
    def __init__(self, dataset1, dataset2, length=None, plotting_path=None) -> None:
        if length is None:
            length = min(len(dataset1), len(dataset2))
        self.data = {}
        for key, value in dataset1.data.items():
            self.data[key]=value[:length]
        for key, value in dataset2.data.items():
            self.data[key]=value[:length]
        self.list_order = dataset1.list_order+dataset2.list_order
        if plotting_path is not None:
            self.plot(plotting_path)