from .scRNADataset import scRNADataset
from vae_dataset import VaeDataset



def create_dataset(dataset_type: str, *args: Any, **kwargs: Any) -> VaeDataset:
    if dataset_type == "bdp": #TODO ANDREI FIX THE NAMES
        return scRNADataset(*args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'.")
