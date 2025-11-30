Instructions  
1. Model 1 and Model 2 are two independent models trained in this study. After entering each model’s directory, run 'Step1, Step2, Step3' in order:  
&nbsp;&nbsp;&nbsp;&nbsp;Step 1: Use the pretrained model to extract the ASTFs of target earthquakes.  
&nbsp;&nbsp;&nbsp;&nbsp;Step 2: Generate plots and compute CCs and misfits to evaluate the accuracy of the ASTFs outputs.  
&nbsp;&nbsp;&nbsp;&nbsp;Step 3: Apply the second seismic moment method to invert the finite source parameters. (This step requires CVX.)
2. Documentation for Second Seismic Moment (Step 3) Inversion  
CVX is a MATLAB-based software for disciplined convex programming, used to solve convex optimization problems in a simple, readable form.  
(1)  Download the CVX installation package from the official website (https://cvxr.com/cvx/download/). Making sure to select the version that matches your operating system (Linux, macOS, or Windows).  
(2)  Extract the package to any folder of your choice. This will create a folder named cvx.  
(3)  Open MATLAB and add the CVX folder to your MATLAB path.  
(4)  In the MATLAB command window, run:  
&nbsp;&nbsp;&nbsp;&nbsp;cd ./cvx  &nbsp;&nbsp;&nbsp;&nbsp; % Please replace with the actual path to your CVX folder  
&nbsp;&nbsp;&nbsp;&nbsp;cvx_setup  
The ‘cvx_setup’ function performs a series of checks and configurations to ensure that the installation is correct.  
