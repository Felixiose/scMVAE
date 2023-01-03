# Mixed Curvature VAE for single-cell RNA sequencing data


by
Colin Doumont,
Christophe Muller,
Andrei Papkou,
and
Félix Vittori


> This repository contains the implementation of our Semester Project for the class Deep Learning 263-3210-00L at ETH Zurich.
> To run this project follow the steps layed out below


## Installation & Configuration

Make sure you have Python 3.7 installed. If you do not want to install all dependencies
manually, make sure to have conda, and run the following commands:
```bash
make conda
conda activate pt
make download_data
```

## Structure 

Our structure is closely related to the one used in the
[MVAE-Github](https://github.com/oskopek/mvae) as it is the one we built our project upon. 





* **`Semster_Project/`** - Source folder 
  * **`data/`** - Data loading, preprocessing, batching, and pre-trained embeddings.
  * **`examples/`** - Contains the main executable file. Reads flags and runs the corresponding training and/or evaluation. 
  * **`scMVAE/`** - folder containing files for the model definition
    * `components/`- contains the components for training
    * `distributions/` - contains different spaces
    *  ** `kNN/` ** - contains the code for clustering and evaluation
      * **`distances.py`** - contains the definitions of the different distances
      * ** `kNN.py` ** - main loop for clustering and evaluation
    * **`model/`** - 
      * **`ffn_vae.py`** - simple feedforward network with one recurrent branch passing the batch effect
      * `train.py` - class for training the model
      * `vae.py` - class for we inherit from in ffn_vae.py
    * `ops/` - operations definitions from mvae
    * `sampling/`- sampling methods from mvae
    * `utils/` - different data handling utils
  * `visualization/` - Utilities for visualization of latent spaces or training statistics.
  * `utils.py/` - Containt parsing utility function
* `data/` - Data folder. Contains a script necessary for downloading the datasets we used. 
* `scripts/` - Contains scripts to run experiments and plot the results. _From MVAE_
* **`Makefile`** - Defines "aliases" for various tasks.
* **`README.md`** - This manual.
* **`environment.yml`** - Required Python packages.


In bold are files that were changed or created by us. The rest of the script is from the MVAE script.

## Usage

To run training and inference, activate the created conda environment and run the examples:

```bash
conda activate pt

python -m Semester_Project.examples.run
```

Take a look at `Semester_Project/examples/run.py` for a list of command line arguments.

For an evaluation run, see `Semester_Project/examples/eval.py`.



# References 


...................

# Contact us

[Colin Doumont](mailto:cdoumont@student.ethz.ch)\
[Christophe Muller](mailto:mullec@student.ethz.ch)\
[Andrei Papkou](mailto:andrei.papkou@uzh.ch)\
[Félix Vittori](mailto:fvittori@student.ethz.ch)
