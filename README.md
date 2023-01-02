# Mixed-Curvature-VAE-for-scRNA-seq


by
Colin Diumont,
Christophe Muller,
Andrei Papkou,
and
Félix Vittori


> This repository contains the implementation of our Semester Project for the class Deep Learning 263-3210-00L at ETH Zurich.
> For further information about this project visit https://github.com/
> To run this project follow the steps layed out below


## Installation & Configuration

Make sure you have Python 3.7 installed. If you do not want to install all dependencies
manually, make sure to have conda, and run the following commands:
'''bash
make conda
conda activate pt
make download_data
'''

## Structure 

Our structure is closely related to the one used in the
mvae-github-repository as it is the one we built our project upon. 
Check mvae out: https://github.com/oskopek/mvae

* `chkpt/` - Checkpoints for trained models.
* `data/` - Data folder. Contains a script necessary for downloading the datasets, and the downloaded data.
* `mt/` - Source folder (stands for Master Thesis).
  * `data/` - Data loading, preprocessing, batching, and pre-trained embeddings.
  * `examples/` - Contains the main executable file. Reads flags and runs the corresponding training and/or evaluation.
  * `mvae/` - Model directory. Note that models heavily use inheritance!
  * `test_data/` - Data used for testing.
  * `visualization/` - Utilities for visualization of latent spaces or training statistics.
* `plots/` - Folder to store generated plots.
* `scripts/` - Contains scripts to run experiments and plot the results.
* `Makefile` - Defines "aliases" for various tasks.
* `README.md` - This manual.
* `LICENSE` - Apache Standard License 2.0.
* `environment.yml` - Required Python packages.

## Usage

To run training and inference, activate the created conda environment and run the examples:

```bash
conda activate pt

python -m mt.examples.run

Take a look at `Semester_Project/examples/run.py` for a list of command line arguments.

For an evaluation run, see `Semester_Project/examples/eval.py`.



# References 


...................