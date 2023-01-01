import typing
import os
from Semester_Project.data.scRNADataset import scRNADataset
from Semester_Project.data.vae_dataset import VaeDataset



#create_dataset(dataset_type="calegans",batch_size=100,)



def create_dataset(dataset_type: str, *args, **kwargs) -> VaeDataset:
    if dataset_type == "adipose":
        return scRNADataset(
                   data_folder = None, 
                   data_file = "./Semester_Project/data/adipose/adipose.mtx", 
                   label_file = "./Semester_Project/data/adipose/adipose_celltype.tsv",
                   batch_files = [],
                   **kwargs
                   )
    elif dataset_type == "adipose_gaussian":
        return scRNADataset(
                   data_folder = None, 
                   data_file = "./Semester_Project/data/adipose_gaussian/adipose.mtx", 
                   label_file = "./Semester_Project/data/adipose_gaussian/adipose_celltype.tsv",
                   batch_files = [],
                   **kwargs
                   )
    elif dataset_type == "rgc":
        return scRNADataset(
                   data_folder = None, 
                   data_file = "./Semester_Project/data/rgc/gene_sorted-rgc.mtx", 
                   label_file = "./Semester_Project/data/rgc/rgc_celltype.tsv",
                   batch_files = [ "./Semester_Project/data/rgc/rgc_batch.tsv" ],
                   **kwargs
                   )
    elif dataset_type == "celegans":
        return scRNADataset(
                   data_folder = None, 
                   data_file = "./Semester_Project/data/celegans/celegan.mtx", 
                   label_file = "./Semester_Project/data/celegans/celegan_embryo_time.tsv",
                   batch_files =[ "./Semester_Project/data/celegans/celegan_batch.tsv",
                   "./Semester_Project/data/celegans/celegan_embryo_time.tsv"],
                   **kwargs
                   )
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_type}'")
