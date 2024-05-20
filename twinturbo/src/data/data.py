import pandas as pd
import h5py
import operator
import numpy as np
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
### Collection of functions for data pre-processing
##############################################

def apply_cuts(self, data: pd.DataFrame, cuts) -> pd.DataFrame:
    for cut in cuts:
        cut = cut.split(" ")
        data = data[OPERATORS[cut[1]](data[cut[0]], float(cut[2]))]

##############################################


class InMemoryDataset():
    """Dataset for loading file(s) into memory, preprocessing them.

    Also fits a sampler for generating in the signal region.
    """

    def __init__(self, file_path: str,) -> None:
        self.file_path = file_path
        self.data = self.load(file_path)
        self.data = [group.values for group in self.data]
        
    def __getitem__(self, item):
        return [group[item] for group in self.data]
    
    def __len__(self):
        return len(self.data[0])
    
    def get_dim(self):
        return [group[0].shape for group in self.data]
    
    def load(self, file_path: str) -> pd.DataFrame:
        with h5py.File(file_path, 'r') as f:
            return [pd.DataFrame(f['data'][:])]
