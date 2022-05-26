# pytorch_dlr

## Pytorch code for convolutional neural network experiments

### Installation

1. create a python virtual environment (pyenv or conda) and install pip using  ``conda install pip``. If you are using no virtual environment, please be aware of
   version incompatibilities of tensorflow.
2. Install the project requirements (example for pip):
   ``pip install -r requirements.txt``
3. Run the batch scripts for the test cases
    1. ``sh produce_results.sh`` or ``bash produce_results.sh`` to reproduce the results of table 6 in the supplementary material 
    2. If you want to rerun the tests of table 6 (supplementary), please use ``sh run_Lenet5_experiment.sh`` or ``bash run_Lenet5_experiment.sh``. It performs 5 runs with different splits of the dataset and different starting points using the DLRT with Lenet on MNIST.
    3. If the shell is not able to run the script in the new enviroment, please try to modify the ``file_name.sh`` files changing ``python3 file_name.sh`` with ``python file_name.sh``
