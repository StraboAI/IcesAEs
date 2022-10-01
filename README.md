Ice AEs
==============================

Classify waveforms from acoustic events emitted by frictional sliding experiments of ice on different substrates.

Code to preprocess data, train, and evaluate machine learning models described in Saltiel et al., 2022 (submitted).

Create a conda environment and install the packages listed in `conda-env.yaml`.

Download the raw waveform data from the figshare site and save in the `data` directory.

Run the notebooks in the following order to reproduce the data used for the paper:

1. preprocess_waveforms.ipynb
2. train_and_evaluate.ipynb
3. create_figures.ipynb

Results are saved in the `data` directory, and figures are saved in the `figures` directory.
