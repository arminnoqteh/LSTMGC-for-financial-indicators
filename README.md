# LSTMGC Project

This project involves training a model using LSTMGC (Long Short-Term Memory Graph Convolution). The model is built using Keras and the graph convolution is implemented using a custom LSTMGC layer.

## Model Details

The model takes as input a sequence of graph-structured data and outputs a forecast for each node in the graph. The model is compiled with the RMSprop optimizer and Mean Squared Error loss function.

## Training

The model is trained on a dataset `train_dataset` and validated on `val_dataset`. The training runs for 24 epochs, or until an early stopping condition is met (no improvement after 10 epochs).

## Model Summary

After training, the model's architecture and parameters can be viewed by calling `model.summary()`.

## Requirements

- Keras
- Custom `graph_conv` module

## Usage

To use this model, you need to provide:

- `input_sequence_length`: The length of the input sequences
- `forecast_horizon`: The number of time steps to forecast
- `graph`: The adjacency matrix of the graph
- `train_dataset` and `val_dataset`: The training and validation datasets

## Future Work

This is a basic implementation and there are many potential improvements and extensions. For example, different graph convolution parameters could be explored, or the model could be adapted to handle multi-feature inputs.
