from .scRNADataset import scRNADataset
from vae_dataset import VaeDataset





def create_dataset(dataset_name: str, *args: Any, **kwargs: Any) -> VaeDataset:
    data_scheme_json = ".data/" + dataset_name + "/"+ dataset_name +".json"
    if os.path.exist(data_scheme_json):
        data_scheme = json.load(open(data_scheme_json, "r"))
        return scRNADataset(batch_size = BATCH_SIZE,
                   data_folder = os.path.dirname(data_scheme['data_file']), 
                   data_file = data_scheme['data_file'], 
                   label_file = data_scheme['label_file'],
                   batch_files = data_scheme['batch_files'],
                   doubles = DOUBLES)
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'.\nMake sure data is in data directory and data scheme json file exists")
   
    
   
       
