# CI-Neural-Network-Project
# CIFAR-10 Image Classification

This project aims to develop and analyze image classification models for the CIFAR-10 dataset using various techniques, including Convolutional Neural Networks (CNNs) and custom neural networks with backpropagation. The goal is to achieve high accuracy in classifying images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes.

## Table of Contents

- [Getting Started](#getting-started)
  - [Dataset](#dataset)
  - [Dependencies](#dependencies)
  - [Code Structure](#code-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)

## Getting Started

### Dataset

1. **Dataset Download**: To replicate the experiments, you need to download the CIFAR-10 dataset. Use the following script to download and extract the dataset:
   ```bash
   !gdown --id 1Y1vgzPvMeVcXSxDfOlCVia7wsU7p8M6g -O CIFAR10.tar.gz
   !tar xzf CIFAR10.tar.gz
   ```

2. **Dataset Details**: The CIFAR-10 dataset consists of 60,000 images divided into a training set of 50,000 images and a test set of 10,000 images. Images are categorized into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

### Dependencies

1. **Library Installation**: Install the required libraries using the following command:
   ```bash
   pip install tensorflow matplotlib
   ```

2. **Library Details**:
   - **TensorFlow**: Used for building and training neural networks.
   - **Matplotlib**: Used for visualizing training results and graphs.

### Code Structure

The project code is organized into the following sections:

1. **Data Preparation**: Load and preprocess the CIFAR-10 dataset. Explore the dataset by displaying sample images from each class.

2. **CNN Model Building**: Construct a CNN-based model using Keras with convolutional and pooling layers to learn feature representations from the images.

3. **CNN Model Training**: Compile and train the CNN model on the CIFAR-10 dataset. Monitor and visualize training and validation accuracy, as well as loss curves.

4. **Custom Neural Network with Backpropagation**: Implement a custom neural network from scratch, focusing on backpropagation for training. Evaluate the network's accuracy.

5. **Optimized Backpropagation**: Improve the backpropagation algorithm using vectorization techniques for enhanced training speed and efficiency.

## Usage

1. Open a Jupyter Notebook or your preferred Python environment.

2. Copy and paste the code sections from the provided notebook into your environment.

3. Run each code section sequentially.

4. Observe the results, including model training accuracy, validation accuracy, loss curves, and backpropagation results.

## Results

This project showcases the evolution of image classification models using different techniques:

- Utilizing a CNN-based model with convolutional and pooling layers for feature extraction.
- Developing a custom neural network from scratch and implementing backpropagation for training.
- Incorporating optimization and vectorization techniques to accelerate the backpropagation algorithm.

Visualizations, such as accuracy and loss curves, offer insights into the model's performance and training progress.

## Contributors

We invite contributions from the community to enhance and refine the CIFAR-10 Image Classification
- Report bugs, suggest improvements, or request features by creating an issue.
- Fork the repository, implement changes, and submit pull requests.
- Enhance fuzzy logic rules, membership functions, and the user interface.


