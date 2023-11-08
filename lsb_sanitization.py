import numpy as np
import onnx
import struct
from onnx import numpy_helper
import numpy as np



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


def compare_graph_structure(original_model, modified_model):
    # Check if the nodes in both graphs have the same structure
    if len(original_model.graph.node) != len(modified_model.graph.node):
        return False

    for original_node, modified_node in zip(original_model.graph.node, modified_model.graph.node):
        if original_node.op_type != modified_node.op_type:
            return False
        if original_node.input != modified_node.input:
            return False
        if original_node.output != modified_node.output:
            return False

    return True


def sanitize_params(model, n_lsbs):
    modified_model = modify_and_update_data(model, n_lsbs)
    return modified_model

# Function to convert float to binary representation
def float_to_binary(value):
    """ Assumes a float32 or float64 value """
    if value.dtype == np.float64:
        return format(struct.unpack('!Q', struct.pack('!d', value))[0], '064b')
    elif value.dtype == np.float32:
        return format(struct.unpack('!I', struct.pack('!f', value))[0], '032b')
    else:
        raise TypeError("Unsupported type, only float32 and float64 are supported")

# Function to modify the least significant bits of a float value
def modify_lsb(binary_str, n_lsbs):
    """ Modify the least significant bits of a binary string """
    return binary_str[:-n_lsbs] + '0' * n_lsbs

# Function to convert binary representation back to float
def binary_to_float(binary_str, dtype):
    """ Assumes a binary string representing a float32 or float64 """
    if dtype == np.float64:
        return struct.unpack('!d', struct.pack('!Q', int(binary_str, 2)))[0]
    elif dtype == np.float32:
        return struct.unpack('!f', struct.pack('!I', int(binary_str, 2)))[0]
    else:
        raise TypeError("Unsupported type, only float32 and float64 are supported")

# Function to modify only float parameters of a model
def modify_float_params(model, n_lsbs):
    """
    Check if the data is stored as raw_data or directly as numeric data (otherwise attacker can bypass the defense)
    Check if the data is stored as float
    Apply LSB modification to the float values
    """
    for initializer in model.graph.initializer:
        # Check if the data is in raw format
        if initializer.HasField('raw_data'):
            data = numpy_helper.to_array(initializer)
        else:
            # Assume the data is in float format
            data = np.array(initializer.float_data).reshape(initializer.dims)
        if data.dtype == np.float32 or data.dtype == np.float64:
            # Flatten the data for individual modifications
            flat_data = data.flatten()
            modified_flat_data = np.array([
                binary_to_float(modify_lsb(float_to_binary(val), n_lsbs), data.dtype)
                for val in flat_data
            ], dtype=data.dtype).reshape(data.shape)
            # Replace the initializer data with modified data
            initializer.CopyFrom(numpy_helper.from_array(modified_flat_data, initializer.name))
    return model

# Function to modify the least significant bits of an integer value
def modify_lsb_for_integers(value, n_lsbs):
    """
    Modify up to n_lsbs least significant bits of an integer to zero,
    without changing more bits than the integer has.
    """
    # Convert the NumPy int64 to native Python int before calling bit_length
    bits_needed = int(value).bit_length()

    # Determine the number of bits we can actually modify
    bits_to_modify = min(n_lsbs, bits_needed)

    # Create a mask for the bits to modify
    mask = (1 << bits_to_modify) - 1

    # Apply the mask to zero out the LSBs
    modified_value = value & ~mask

    return modified_value


# Function to apply LSB modification to integer initializers in the model
def modify_integer_params(model, n_lsbs):
    """
    Check if the data is stored as raw_data or directly as numeric data (otherwise attacker can bypass the defense)
    Check if the data is of integer type
    Apply LSB modification to the integer values
    """
    for initializer in model.graph.initializer:
        # Check if the data is in raw format
        if initializer.HasField('raw_data'):
            data = numpy_helper.to_array(initializer)
        else:
            # Assume the data is in float format
            data = np.array(initializer.float_data).reshape(initializer.dims)
        # Check if the data is of integer type
        if np.issubdtype(data.dtype, np.integer):
            # Apply the LSB modification
            flat_data = data.flatten()
            # Modify this line in the list comprehension
            modified_flat_data = np.array([modify_lsb_for_integers(int(val), n_lsbs) for val in flat_data], dtype=data.dtype).reshape(data.shape)
            # Replace the initializer data with modified data
            initializer.CopyFrom(numpy_helper.from_array(modified_flat_data, initializer.name))
    return model

def sanitize_model(model, n_lsbs):
    """
    Take an ONNX model, modify its numeric parameters' LSBs, and save to a new file.
    """
    modified_model_float = modify_float_params(model, n_lsbs)
    modified_model_full = modify_integer_params(modified_model_float, n_lsbs)
    return modified_model_full

