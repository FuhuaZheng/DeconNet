Instructions

1. Model 1 and Model 2

Model 1 and Model 2 are two independent models trained in this study. After entering each model’s directory, run step1, step2, and step3 in order:

- Step 1: Use the pretrained model to extract the ASTFs of earthquake events.
- Step 2: Generate plots and compute CCs and misfits to evaluate the accuracy of the ASTF outputs.
- Step 3: Apply the second seismic moment method to invert the finite source parameters.

2. Documentation for Second Seismic Moment Inversion

(1)  Download the CVX installation package from the official website (https://cvxr.com/cvx/download/).
(2)  Extract the package to any folder of your choice. This will create a folder named cvx.
(3)  Open MATLAB and add the CVX folder to your MATLAB path.
(4)  In the MATLAB command window, run:
    cd ./cvx   % replace with the actual path to your CVX folder
    cvx_setup
The cvx_setup function performs a series of checks and configurations to ensure that the installation is correct.
