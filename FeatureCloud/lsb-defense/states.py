from FeatureCloud.app.engine.app import AppState, app_state, Role
from lsb_sanitization import sanitize_params, compare_original_to_modified, sanitize_model, compare_graph_structure
import onnx
import os
import bios

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.

INPUT_DIR = 'mnt/input/'
OUTPUT_DIR = 'mnt/output/'

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.

@app_state('initial')
class InitialState(AppState):
    def register(self):
        self.register_transition('read_input', Role.COORDINATOR)
        self.register_transition('output', Role.PARTICIPANT)
        #self.register_transition('terminal')  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.update(progress=0.1)
        if self.is_coordinator:
            return 'read_input'
        else:
            return 'output'

    #return 'terminal'  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.

#a state for reading inputs
@app_state('read_input')
class InputState(AppState):
    def register(self):
        self.register_transition('sanitize', Role.COORDINATOR)

    def run(self):
        self.update(progress=0.2)
        self.read_config()
        self.get_model()
        self.update(progress=0.3)
        self.log('Config file and input model read successfully')
        return 'sanitize'

    def read_config(self):
        config = bios.read(INPUT_DIR + 'config.yml')
        self.store('model_name', config['model_name'])
        self.store('n_lsbs', config['n_lsbs'])

    def get_model(self):
        model_name = self.load('model_name')
        model_path = os.path.join(os.getcwd(), INPUT_DIR, f'{model_name}.onnx')
        #model = load_pytorch_from_onnx(model_path)
        original_model = onnx.load(model_path)
        # Data can be stored for access in other states like this
        self.store(key="original_model", value=original_model)


# a state for the local computation
@app_state('sanitize')
class sanitization(AppState):
  def register(self):
    self.register_transition("output", Role.COORDINATOR)


  def run(self):
      self.update(progress=0.4)
      modified_model = sanitize_model(self.load("original_model"), self.load("n_lsbs"))
      self.update(progress=0.5)
      self.log('Model parameters sanitized successfully')
      self.store(key="modified_model", value=modified_model)
      self.update(progress=0.6)
      structure = compare_original_to_modified(self.load("original_model"), modified_model)
      if structure == True:
          self.log('The structure of the original and modified models are the same')
      elif structure == False:
          self.log('Error during sanitization: The structure of the original and modified models are not the same')
      comp_graph_structure = compare_graph_structure(self.load("original_model"), modified_model)
      if comp_graph_structure == True:
          self.log('The computational graph str. of the models are the same')
      elif comp_graph_structure == False:
          self.log('Error during sanitization: The computational graph str. models are not the same')
      self.update(progress=0.7)
      return 'output'


@app_state('output')
class OutputState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        self.update(progress=0.8)
        modified_model = self.load('modified_model')
        self.update(progress=0.9)
        onnx.save(modified_model, f'{OUTPUT_DIR}modified_model.onnx')
        n_lsbs = self.load('n_lsbs')
        self.update(progress=1)
        self.log(f'The {n_lsbs} LSBs of model parameters were sanitized')
        return 'terminal'
