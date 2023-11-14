# FeatureCloud Parameter Sanitization App

The app implements a defense against a white-box data exfiltration attack by sanitizing the given number of least significant bits of trainable ML model parameters.  
The information hidden in the model parameters will be modified or removed by sanitizing the least significant bits of the parameters.  
The sanitized model can be used for inference. It is important to note that applying this defense might compromise the original performance of the model.  
Furthermore, this approach can potentially remove a watermark embedded in the model parameters.
The app takes a model as an input as returns a defended model.

# Usage
## Input data - Client's data
This app is designed for a single-client/coordinator usage. A client's data folder should contain the following files:
 - **config**.yml: the configuration file of the app [`config.yml`]
 - **model** to be sanitized in onnx format, e.g. [`model.onnx`]

### Config file
This file contains the hyperparameters that need to be provided by the client for the execution of the app:
Following information should be provided in the config file:
 - **n_lsbs**: the number of least significant bits of the trainable parameters to be sanitized. If the n_lsbs is larger than the number of bits needed to represent a parameter, all bits of the parameter will be sanitized.
 - **model_name**: the name of the model in the input folder to be sanitized (e.g. if the file is cnn.onnx, the name is "cnn")


The required information should be provided in the following form inside a .yml file, e.g.:  
model_name: "model"  
n_lsbs: 10  

## AppStates
This app implements four states  
    - [`initial`]: The app is initialized  
    - [`read_input`]: The app reads the input config file and the model  
    - [`sanitize`]: The app determines in which representation the parameters are saved within the onnx file to prevent an attack from circumventing this defense. Subsequently, it sanitizes the model and checks the structure of the original and the modified model to ensure the validity of the sanitized model.  
    - [`output`]: The app returns and saves the sanitized model  


## Output data
The app returns the sanitized model in onnx format, i.e. [`sanitized_model.onnx`].
The sanitized model can be used for further inference.


### This app is a part of the FeatureCloud AI Store:
<a id="1">[1]</a> 
Matschinske, J., Späth, J., Nasirigerdeh, R., Torkzadehmahani, R., Hartebrodt, A., Orbán, B., Fejér, S., Zolotareva,
O., Bakhtiari, M., Bihari, B. and Bloice, M., 2021.
The FeatureCloud AI Store for Federated Learning in Biomedicine and Beyond. arXiv preprint arXiv:2105.05734.
