# Mixed Curvature VAE for single-cell RNA sequencing data


by
Colin Doumont,
Christophe Muller,
Andrei Papkou,
and
Félix Vittori


> This repository contains the implementation of our Semester Project for the class Deep Learning 263-3210-00L at ETH Zurich.
> To run this project follow the steps layed out below.


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


* **`Semester_Project/`** - Source folder 
  * **`data/`** - Data loading, preprocessing, batching, and pre-trained embeddings.
  * **`examples/`** - Contains the main executable file. Reads flags and runs the corresponding training and/or evaluation. 
  * **`scMVAE/`** -  Contains all files needed for the model definition.
    * `components/`- Contains the components needed for training.
    * `distributions/` - Contains the probability distributions for the different spaces.
    * **`kNN/`** - Contains the code for the kNN clustering and silhouette scores.
      * **`kNN.py/`** - Runs the kNN algorithm on the whole dataset.
      * **`kNN_samples.py/`** - Runs the kNN algorithm on subsamples of the whole dataset (faster).
      * **`silhouette_samples.py/`** - Computes the silhouette scores with regards to the batch effects.
    * **`model/`** - 
      * **`ffn_vae.py`** - Simple feedforward network with one recurrent branch passing the batch effect.
      * `train.py` - Class for training the model.
      * `vae.py` - Class inherited by ffn_vae.py/
    * `ops/` - Contains the operations definitions from mvae.
    * `sampling/`- Contains the sampling methods from mvae.
    * `utils/` - Contains different data handling utils.
  * `visualization/` - Utilities for visualization of latent spaces or training statistics.
  * `utils.py/` - Contains parsing utility function.
* `data/` - Data folder. Contains a script necessary for downloading the datasets we used. 
* **`scripts/`** - Contains scripts to run experiments presented in paper.
* **`Makefile`** - Defines "aliases" for various tasks.
* **`README.md`** - This manual.
* **`environment.yml`** - Required Python packages.


In bold are files that were changed or created by us. The rest of the script is from the MVAE script.

## Usage

To get a feel for how the model works, try out the toy example by running:

```bash
conda activate pt
make run
```
Take a look at `Semester_Project/examples/run.py` for a list of command line arguments.
For an evaluation run, see `Semester_Project/examples/eval.py`.

## Replication of experiments

To replicate our experiments step by step, please have a look at the `scripts/` folder.






## Contact us

[Colin Doumont](mailto:cdoumont@student.ethz.ch)\
[Christophe Muller](mailto:mullec@student.ethz.ch)\
[Andrei Papkou](mailto:andrei.papkou@uzh.ch)\
[Félix Vittori](mailto:fvittori@student.ethz.ch)
