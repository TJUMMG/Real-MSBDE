# Real-MSBDE: Real-World Image Bit-Depth Enhancement
This repository provides two models (the classification-based degradation model and the bit-depth enhancement model Real-MSBDE) of "Real-World Image Bit-Depth Enhancement".
- If you use this code, please cite the following publication: Y.Gang, J.Liu, P.Jing, H.Duan, and G.Zhai, "Real-World Image Bit-Depth Enhancement".

## Environment
Our model is tested on Ubuntu with the following environments:
- python==3.6.13
- pytorch==1.8.0
- tensorflow==1.9.0

The complete environment can be found in `environment.yml`.


## Dataset
We provide four samples for each of the two models for testing, located in `./degradation/dataset` and `./Real-MSBDE/dataset`.
- Note: this is real-world image BDE, so all datasets provided, are real 8-bit and real 16-bit.

## Models
The weights of two models are located in `./degradation/weights` and `./Real-MSBDE/weights`.

## Test
For the classification-based degradation model, the model input is the quantization part and the residual part of the HBD image, so the HBD image is first divided into two parts by running
```
degradation
- to_res.py
```

Then, the LBD image is obtained by running
```
degradation
- test.py
```

For the bit-depth enhancement model Real-MSBDE, you can run
```
Real-MSBDE
- main.py
```

You can find the outputs of two models in `./degradation/test_results` and `./Real-MSBDE/test_results`



## Acknowledgement
- Thanks to *Sung-Jin Cho, et al.*, who are authors of "Rethinking coarse-to-fine approach in single image deblurring", published in the Proceedings of the IEEE/CVF international conference on computer vision, for referring to their outstanding work.
