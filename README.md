# Project Title

This project is a collection of Python scripts that demonstrate various functionalities related to machine learning, specifically deep learning using PyTorch. The scripts cover a range of topics from data preprocessing, model creation, training, testing, and visualization of results.

## Files Description

- `nn_seq.py`: This script defines a convolutional neural network model using PyTorch's Sequential API.

- `read_data.py`: This script reads image data from a directory and creates a PyTorch Dataset object.

- `rename_dataset.py`: This script renames images in a dataset and writes the labels to a text file.

- `test.py`: This script loads a pre-trained model and uses it to make predictions on a single image.

- `test_tensorboard.py`: This script demonstrates how to use TensorBoard for visualization of model training.

- `train.py`: This script trains a model on the CIFAR-10 dataset and saves the model after each epoch.

- `train_gpu_1.py` and `train_gpu_2.py`: These scripts are variations of `train.py` that demonstrate how to use a GPU for training if available.

- `transforms.py` and `transformsV2.py`: These scripts demonstrate how to use PyTorch's transforms for data augmentation.

- `nn_relu.py`: This script demonstrates the use of ReLU activation function in a neural network.

- `nn_module.py`: This script demonstrates the basic structure of a PyTorch Module.

- `nn_maxpool.py`: This script demonstrates the use of MaxPooling in a convolutional neural network.

- `nn_optim.py`: This script demonstrates how to use PyTorch's SGD optimizer.

- `nn_loss_network.py`: This script demonstrates how to calculate the loss of a network using CrossEntropyLoss.

- `nn_linear.py`: This script demonstrates the use of a Linear layer in a neural network.

- `nn_loss.py`: This script demonstrates how to calculate different types of loss functions in PyTorch.

- `nn_conv2d.py`: This script demonstrates the use of a Conv2D layer in a convolutional neural network.

- `nn_conv.py`: This script demonstrates how to perform a 2D convolution operation using PyTorch's functional API.

## Requirements

- Python 3.6 or above
- PyTorch 1.0 or above
- torchvision
- PIL
- TensorBoard

## Usage

Each script can be run independently. For example, to run `nn_seq.py`, use the following command:

```bash
python nn_seq.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)