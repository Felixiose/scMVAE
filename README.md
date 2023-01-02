# Mixed Curvature VAE for single-cell RNA sequencing data


by
Colin Doumont,
Christophe Muller,
Andrei Papkou,
and
Félix Vittori


> This repository contains the implementation of our Semester Project for the class Deep Learning 263-3210-00L at ETH Zurich.
> For further information about this project visit https://github.com/.
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





* `Semster_Project/` - Source folder 
  * `data/` - Data loading, preprocessing, batching, and pre-trained embeddings.
  * `examples/` - Contains the main executable file. Reads flags and runs the corresponding training and/or evaluation. _changed to our needs from mvae_
  * `scMVAE/` - Model directory. Note that models heavily use inheritance!
    * `components/`- from mvae, contains the components for training
    * `distributions/` - from mvae, contains different spaces
    *  `kNN/` - contains the code for clustering 
    * `model/` - contains the ffnn_vae model (our contribution) and the vae.py object from which we inherit the model. In addition we have a training class, also from mvae
    * `ops/` - operations definitions from mvae
    * `sampling/`- sampling methods from mvae
    * `utils/` - different data handling utils
  * `visualization/` - Utilities for visualization of latent spaces or training statistics.
  * `utils.py/` - Containt parsing utility function
* `data/` - Data folder. Contains a script necessary for downloading the datasets we used. 
* `scripts/` - Contains scripts to run experiments and plot the results. _From MVAE_
* `Makefile` - Defines "aliases" for various tasks.
* `README.md` - This manual.
* `environment.yml` - Required Python packages.

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

[Colin Doumont](cdoumont@student.ethz.ch)\
[Christophe Muller](mullec@student.ethz.ch)\
[Andrei Papkou](andrei.papkou@uzh.ch)\
[Félix Vittori](fvittori@student.ethz.ch)
