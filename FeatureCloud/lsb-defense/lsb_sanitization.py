import numpy as np
import onnx
import struct
from onnx import numpy_helper
import numpy as np






#def float2bin32(f):
#    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f))

# Function to convert float to binary
def float_to_binary32(value):
    return format(struct.unpack('!I', struct.pack('!f', value))[0], '032b')

# Function to convert binary to float
def binary32_to_float(binary_str):
    return struct.unpack('!f', struct.pack('!I', int(binary_str, 2)))[0]

# Function to modify the least significant bits
def modify_lsb(value, n_lsbs):
    binary_value = float_to_binary32(value)
    modified_binary = binary_value[:-n_lsbs] + '0' * n_lsbs
    return binary32_to_float(modified_binary)

#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters())


"""
def params_to_bits(params):
    params_as_bits = []
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            flattened_value = value.flatten()  # flatten the tensor
            for v in flattened_value:
                params_as_bits.extend(float2bin32(v))
    params_as_bits = ''.join(params_as_bits)
    return params_as_bits

def prepare_params(params):
    # ==========================================================================================
    # PREPARE PARAMETERS
    # ==========================================================================================
    #get the shape of the parameters - Each key in the dictionary should be a string representing the name of a parameter,
    #and each value should be a tuple representing the shape of the corresponding tensor.
    params_shape_dict = {}
    for key, value in params.items():
        params_shape_dict[key] = value.shape
    #convert the parameters to bits
    params_as_bits = params_to_bits(params)
    # ==========================================================================================
    return params_as_bits, params_shape_dict
"""

# Function to modify the least significant bits and update raw_data
def modify_and_update_data(model, n_lsbs):
    for initializer in model.graph.initializer:
        # Check if the data is in raw format
        if initializer.HasField('raw_data'):
            weight_data = numpy_helper.to_array(initializer)
        else:
            # Assume the data is in float format
            weight_data = np.array(initializer.float_data).reshape(initializer.dims)

        if n_lsbs > 0:
            # Flatten the weight data for easier processing
            flattened_weights = weight_data.flatten()

            # Modify the n_lsbs of each float value
            modified_weights = [modify_lsb(value, n_lsbs) for value in flattened_weights]

            # Reshape the modified weights to the original shape
            modified_weights = np.array(modified_weights).reshape(weight_data.shape)

            # Convert the modified weights to the serialized binary format
            serialized_data = modified_weights.astype(np.float32).tobytes()

            # Clear the existing data fields
            initializer.ClearField('float_data')

            # Update the raw_data field of the initializer
            initializer.raw_data = serialized_data

    return model


def compare_original_to_modified(original_model, modified_model):
    # Load the original and modified ONNX models

    # Extract initializers (parameters) from both models
    original_initializers = {init.name: numpy_helper.to_array(init) for init in original_model.graph.initializer}
    modified_initializers = {init.name: numpy_helper.to_array(init) for init in modified_model.graph.initializer}

    # Check if the initializers in both models have the same structure
    same_structure = True
    if set(original_initializers.keys()) != set(modified_initializers.keys()):
        same_structure = False
    else:
        for key in original_initializers:
            if original_initializers[key].shape != modified_initializers[key].shape:
                same_structure = False
                break

    return same_structure

def sanitize_params(model, n_lsbs):
    modified_model = modify_and_update_data(model, n_lsbs)
    return modified_model
