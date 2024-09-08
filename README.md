# KAN (Kolmogorov-Arnold Networks) Implementation

This repository contains an implementation of Kolmogorov-Arnold Networks (KAN) for digit classification using the MNIST dataset. The implementation includes the network architecture, data loading, training, and evaluation processes.

## Overview

KAN is a type of neural network that utilizes B-spline bases for its layers. This repository provides a complete example of how to define, train, and evaluate a KAN model.

## Project Structure

- `layer.py`: Contains the implementation of the `Layer` class, which is a key component of the KAN network.
- `kan.py`: Defines the `KAN` class, which builds the Kolmogorov-Arnold Network using multiple `Layer` instances.
- `train.py`: Script to train the KAN model using MNIST data.
- `eval.py`: Script to evaluate the trained KAN model on the validation dataset.

## Dependencies

- PyTorch
- torchvision
- numpy
- tqdm

Install the dependencies using:

```bash
pip install torch torchvision numpy tqdm
```

## Implementation Details

### `Layer` Class

The `Layer` class is a fundamental part of the KAN network. It implements a neural network layer using B-spline bases with the following features:

- **Input and Output Features**: Configurable number of input and output features.
- **Grid Size and Spline Order**: Parameters for B-spline basis functions.
- **Scaling Factors**: For noise, base weights, and spline weights.
- **Activation Function**: Default is `SiLU`.
- **Functions**: Includes methods for parameter resetting, spline weight scaling, and forward pass operations.

### `KAN` Class

The `KAN` class defines the Kolmogorov-Arnold Network with the following features:

- **Network Architecture**: Includes hidden layers with sizes [784, 64, 10].
- **Training and Validation**: Uses MNIST dataset for training and validation.
- **Optimizer**: AdamW with a learning rate of `1e-3` and weight decay of `1e-4`.
- **Learning Rate Scheduler**: `ExponentialLR` with a decay factor of `0.8`.
- **Loss Function**: CrossEntropyLoss for classification.

### Data Loading and Training

#### Data Loading

- **Data Preparation**: Data is transformed and normalized using `transforms.Compose`. Normalization scales pixel values to the range [-1, 1], which is suitable for training neural networks.
- **Training and Validation Datasets**: MNIST datasets for training (`train=True`) and validation (`train=False`) are downloaded and loaded.
- **Data Loaders**: `DataLoader` objects are used to load data in batches of 64 for training and validation.

#### Training Process

- **Model Training Mode**: The model is set to training mode (`train()`).
- **Training Loop**: Training data is loaded from `trainloader` and the following operations are performed:
  - **Image Preprocessing**: Images are converted to 784-dimensional vectors and transferred to the computation device.
  - **Gradient Zeroing**: Gradients of the optimizer are zeroed.
  - **Prediction and Error Calculation**: Model output is computed, and error is calculated using the loss function.
  - **Gradient Calculation and Update**: Gradients are calculated and model weights are updated.
  - **Accuracy Calculation**: Model accuracy is computed based on outputs and true labels.
  - **Progress Display**: Training progress is displayed using `tqdm`.

### Evaluation Process

- **Model Evaluation Mode**: The model is set to evaluation mode (`eval()`).
- **Evaluation Loop**: Validation data is loaded from `valloader` and the following operations are performed:
  - **Image Preprocessing**: Images are converted to 784-dimensional vectors and transferred to the computation device.
  - **Prediction and Error Calculation**: Model output is computed, and error is calculated using the loss function.
  - **Accuracy Calculation**: Model accuracy is computed based on outputs and true labels.
  - **Result Collection**: Error and accuracy results are collected and averaged.

## Training Script

Run the training script with:

```bash
python train.py
```

## Evaluation Script

Run the evaluation script with:

```bash
python eval.py
```

## Learning Rate Update and Results Display

- **Learning Rate Update**: The learning rate is updated using the `scheduler`.
- **Results Display**: Model error and accuracy on the validation dataset are printed for each epoch.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The MNIST dataset is provided by Yann LeCun and colleagues.
- PyTorch and torchvision libraries are used for deep learning and data handling.

For any questions or suggestions, please open an issue or pull request in this repository.
```
