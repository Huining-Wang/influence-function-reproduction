# influence-function-reproduction
This repository contains a PyTorch implementation reproducing **Figure 2** (Linear Approximation vs. Actual Loss Change) from the paper *Understanding Black-box Predictions via Influence Functions* (Koh & Liang, 2017). 

## Implementation Details
- **Method**: Primitive PyTorch implementation (Calculated gradients and Hessian-Vector Products using autograd).
- **Optimizer**: L-BFGS .
- **Hessian Inverse**: LiSSA .
- **Dataset**: Binary MNIST (0 vs 1).

## Files
- `figure2_linear_approx.py`: Main script to run the experiment and generate the plot.
- `influence_functions.py`: Core implementation of gradient and HVP calculations.

## Results
The reproduction demonstrates a near-perfect linear correlation between the predicted influence and the actual loss change (verified via Leave-One-Out retraining).

## How to Run
1. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib tqdm
2. Run the reproduction script:
   ```bash
   python figure2_linear_approx.py
