from libs_snap.anomdiff.src.datamodules.cnst_lhco import load_data, train_valid_split, get_cut_mask
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from functools import partial
from typing import Literal, Mapping
from copy import deepcopy
import pandas as pd

class LHCOhlvDatasetTT(Dataset):
    """Only stores the high level features and the mjj."""

    def __init__(
        self,
        bkg_path: str,
        sig_path: str | None = None,
        m_add_path: str | None = None,
        n_bkg: int | None = None,
        n_sig: int | None = None,
        n_csts: int | None = None,
        mjj_window: tuple | list | None = ((2700, 3300), (3700, 6000)),
    ) -> None:
        super().__init__()

        # Load the data
        hlv1, hlv2, _jet1, _jet2, _label, mjj = load_data(
            bkg_path, sig_path, n_bkg, n_sig, 0, mjj_window
        )

        # Add a tiny amount of noise to the input number of constituents (dequant)
        hlv1[:, -1] += np.random.randn(len(hlv1))
        hlv2[:, -1] += np.random.randn(len(hlv2))

        # We are generating the joint high level variabels
        self.hlv = np.concatenate([hlv1, hlv2], axis=-1)
        self.mjj = mjj[..., None]  # Must be Bx1 dimension
        
        mjj_add = pd.read_hdf(m_add_path).to_numpy(np.float32)
        cut = get_cut_mask(mjj_add, [[mjj_window[0][0], mjj_window[1][1]]])
        mjj_add = mjj_add[cut.flatten()]
        np.random.shuffle(mjj_add)
        self.mjj_add = mjj[:len(mjj_add)]

    def __len__(self) -> int:
        return len(self.mjj)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.hlv[index], self.mjj[index], self.mjj_add[index]

class LHCOhlvDatasetTT_export(Dataset):
    """Only stores the high level features and the mjj."""

    def __init__(
        self,
        bkg_path: str,
        sig_path: str | None = None,
        m_add_path: str | None = None,
        n_bkg: int | None = None,
        n_sig: int | None = None,
        n_csts: int | None = None,
        mjj_window: tuple | list | None = ((2700, 3300), (3700, 6000)),
        oversampling: int = 1,
    ) -> None:
        super().__init__()

        # Load the data
        hlv1, hlv2, _jet1, _jet2, _label, mjj = load_data(
            bkg_path, sig_path, n_bkg, n_sig, 0, mjj_window
        )

        # Add a tiny amount of noise to the input number of constituents (dequant)
        hlv1[:, -1] += np.random.randn(len(hlv1))
        hlv2[:, -1] += np.random.randn(len(hlv2))

        # We are generating the joint high level variabels
        self.hlv = np.concatenate([hlv1, hlv2], axis=-1)
        self.mjj = mjj[..., None]  # Must be Bx1 dimension
        
        mjj_add = pd.read_hdf(m_add_path).to_numpy(np.float32)
        cut = get_cut_mask(mjj_add, [[mjj_window[0][0], mjj_window[1][1]]])
        mjj_add = mjj_add[cut.flatten()]
        np.random.seed(1) 
        np.random.shuffle(mjj_add)
        self.mjj_add = mjj[:len(mjj_add)]

    def __len__(self) -> int:
        return len(self.mjj)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.hlv[index], self.mjj[index], self.mjj_add[index]

class LHCOLowDatasetTT(Dataset):
    """Combines the jet information to train on both independantly."""

    def __init__(
        self,
        bkg_path: str,
        sig_path: str | None = None,
        n_bkg: int | None = None,
        n_sig: int | None = None,
        n_csts: int | None = None,
        mjj_window: tuple | list | None = ((2700, 3300), (3700, 6000)),
    ) -> None:
        super().__init__()

        # Load the data
        hlv1, hlv2, jet1, jet2, _label, mjj = load_data(
            bkg_path, sig_path, n_bkg, n_sig, n_csts, mjj_window
        )

        # Combine the leading and subleading jets into one array
        self.hlv = np.concatenate([hlv1, hlv2])
        self.jet = np.concatenate([jet1, jet2])
        self.mask = np.any(self.jet, axis=-1)
        print("###################### jet length", len(self.jet))
        #print("###################### jet1 length", len(self.mask))
        print("###################### mjj length", len(mjj))

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.jet[index], self.mask[index], self.hlv[index], self.hlv[index]
    
class LHCOLowModuleTT(LightningDataModule):
    """Datamodule for the LHCO dataset for training on sidebands."""

    def __init__(
        self,
        *,
        dataset: partial,
        loader_kwargs: Mapping,
        val_frac: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        """Set up the relevant datasets."""

        # Load the full sideband data
        full_set = self.hparams.dataset()

        self.train_set, self.valid_set = train_valid_split(
            full_set, self.hparams.val_frac
        )
        self.test_set = full_set  # For our generative models we give the full set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        val_kwargs = deepcopy(self.hparams.loader_kwargs)
        val_kwargs["drop_last"] = False
        return DataLoader(self.valid_set, **val_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self) -> tuple:
        return [3], [5]

    def get_var_group_list(self):
        return [["constituents"], ["hlv"], ["hlv"]]
    
class LHCOhlvModuleTT(LightningDataModule):
    """Datamodule for the LHCO dataset for training on sidebands."""

    def __init__(
        self,
        *,
        dataset: partial,
        loader_kwargs: Mapping,
        val_frac: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        """Set up the relevant datasets."""

        # Load the full sideband data
        full_set = self.hparams.dataset()

        self.train_set, self.valid_set = train_valid_split(
            full_set, self.hparams.val_frac
        )
        self.test_set = full_set  # For our generative models we give the full set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        val_kwargs = deepcopy(self.hparams.loader_kwargs)
        val_kwargs["drop_last"] = False
        return DataLoader(self.valid_set, **val_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self) -> tuple:
        return [10], [1]

    def get_var_group_list(self):
        return [["pt1", "eta1", "phi1", "mj1", "N_con1", "pt2", "eta2", "phi2", "mj2", "N_con2"], ["mjj"], ["mjj"]]

