DRNv2
DRNv2 Predicting Defocus Distance of FPM Low-Resolution Images via Deep Learning Convolutional Neural Networks

The code for DRN/DRNv2 has been released.

Paper link: 
DRN: https://doi.org/10.1364/OE.512330
DRNv2:Not available

Data: https://doi.org/10.5281/zenodo.13845883

Top-level folder structure:
├── Train64              # Model training set (size 64×64; images of other sizes are in "data")
├── networks             # Model functions for DRN/DRNv2
├── environment.txt      # Anaconda environment
├── train.py             # Main Python training script
├── predict.py           # Script for making predictions after training the model
└── README.md
