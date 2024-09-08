from libs_snap.anomdiff.src.datamodules.cnst_lhco import load_data, train_valid_split
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from functools import partial
from typing import Literal, Mapping
from copy import deepcopy

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
        self.mjj = mjj
        self.jet = np.concatenate([jet1, jet2], axis=1)
        self.mask = np.any(self.jet, axis=-1)
        print("###################### jet length", len(self.jet))
        #print("###################### jet1 length", len(self.mask))
        print("###################### mjj length", len(mjj))

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.jet[index], self.mask[index], self.mjj[index], self.mjj[index]
    
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
        return 3, 1

    def get_var_group_list(self):
        return ["constituents", "mjj", "mjj"]

