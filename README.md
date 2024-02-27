# FeatureCloud Parameter Sanitization App

The app implements a defense against a white-box data exfiltration attack by sanitizing the given number of least significant bits of trainable ML model parameters.


## Description

The information hidden in the model parameters will be modified or removed by sanitizing the least significant bits of the parameters.
The sanitized model can be used for inference. It is important to note that applying this defense might compromise the original performance of the model.
Furthermore, this approach can potentially remove a watermark embedded in the model parameters.
The app takes a model as an input as returns a defended model.

## Input

### Client's data

This app is designed for a single-client/coordinator usage. A client's data folder should contain the following files:

- **config**.yml: the configuration file of the app [`config.yml`]
- **model** to be sanitized in onnx format, e.g. [`model.onnx`]

An example config.yml file and a sample model named "simple_mlp.onnx" is added to the repository for the purpose of testing the app. These files are provided in the "featurecloud-sign-modification-app/data/general_directory". The location of the data folder when testing is determined by the location of where the featurecloud controller is started, therefore you might need to manually create a data folder including the general_directory at the same level as the app directory, and move the config and model file there.

#### Config file

This file contains the hyperparameters that need to be provided by the client for the execution of the app:
Following information should be provided in the config file:

- **n_lsbs**: the number of least significant bits of the trainable parameters to be sanitized. If the n_lsbs is larger than the number of bits needed to represent a parameter, all bits of the parameter will be sanitized.
- **model_name**: the name of the model in the input folder to be sanitized (e.g. if the file is cnn.onnx, the name is "cnn")

The required information should be provided in the following form inside a .yml file, e.g.:
model_name: "model"
n_lsbs: 10

## Output

The app returns the sanitized model in onnx format, i.e. [`sanitized_model.onnx`].
The sanitized model can be used for further inference.

## Workflows

As anoher app is unable to call this app to perform the defense in a federated setting, the defense can be applied as a single-client defense on one model at a time, and can therefore be used e.g. by the aggregator to defend the final, aggregated model.

### AppStates

This app implements four states

- [`initial`]: The app is initialized
- [`read_input`]: The app reads the input config file and the model
- [`sanitize`]: The app determines in which representation the parameters are saved within the onnx file to prevent an attack from circumventing this defense. Subsequently, it sanitizes the model and checks the structure of the original and the modified model to ensure the validity of the sanitized model.
- [`output`]: The app returns and saves the sanitized model

## Config

The required information should be provided in the following form inside a .yml file, e.g.:
model_name: "model"
n_lsbs: 10
