# C++ Torch Inference Demo from TensorFlow Model

1. `convertmodel_tf2torch.py` - Convert a DeepPET Tensorflow model (https://github.com/NuriaRufo/DeepPETmodified) to PyTorch and export the converted model to Torch Script module for libTorch C++ inference.
2. `TorchAppMain.cpp`, `CMakeLists.txt` - An C++ App to read NIfTI sinogram volumne and Torch Script module. Output another NIfTI PET Img Volume.
