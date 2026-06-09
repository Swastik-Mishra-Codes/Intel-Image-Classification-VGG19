# Intel Image Classification via VGG-19 with Custom Bounded Double-Arctan Loss

This repository contains an experimental deep learning project focused on image classification using the VGG-19 architecture.<br>The objective of the project was to evaluate model performance across natural and man-made environments while developing a custom mathematical optimization function to replace standard categorical cross-entropy.

<br>The model utilizes the Intel Image Classification dataset sourced from Kaggle. The dataset contains approximately 25,000 images normalized to 150x150 pixels, distributed across 6 distinct environmental classes:
* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

<br>IWe wanted a custom loss function that punishes classification errors harder than standard cross-entropy. Our first version used a squared-exponential math function, but it instantly caused numerical overflows and broke the training with NaN errors. To fix this, our second version switched to a linear absolute error. While this kept the code stable and stopped the crashing, the mathematical curve became too flat near the end of training, preventing the model from accurately learning the final details for the 6 image classes.<br>Hence, to overcome this we used arctan to restrict the loss and prevent overshooting and multiplied it with the exponential function to increase the scale, but to prevent overshooting we restricted the exponents power also usnig arctan.
<br>

## The loss function formula we used is

$$Loss = \arctan(\text{error}^2) \cdot e^{\arctan(\text{error}^2)} + 10^{-4} \cdot (\text{error}^2)$$

## The Team Contribution

The project was executed through a distributed collaborative workflow designed around an iterative code-generation and testing pipeline.

### Swastik Mishra 
* **Role:** Algorithmic Formulation and Code Synthesis
* **Contributions:** Defined the mathematical constraints for the custom Bounded Double-Arctan loss function. Handled the prompt sequencing and synthesis for the core neural network structure, successfully integrating the custom loss tensor operations into the model training framework.

### Akash Magai
* **Role:** Compute Execution and Local Optimization
* **Contributions:** Managed the local runtime environment and hardware configuration. Successfully executed the extended training runs on local hardware, maintaining continuous compute stability over a 6-hour execution cycle, and captured the final convergence metrics and performance outputs.

### Raushan Mehta
* **Role:** Dataset Preparation and Execution Verification
* **Contributions:** Sourced and downloaded the Intel Image Classification dataset from Kaggle, setting up the local folder structures for the 6 target classes. Verified that the early-stage code blocks loaded the images correctly without errors, and assisted in cross-checking the final accuracy logs after the 6-hour training run completed.
