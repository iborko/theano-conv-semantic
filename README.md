# theano-conv-semantic
*!under development!*

Convolutional deep network framework for semantic segmentation, implemented with Theano.

## Installation
Just do

```
    git clone https://github.com/iborko/theano-conv-semantic
```

Later, you can update your repo with

```
    git pull
```

## Description
This framework is heavily based on the work of [Farabet-Pami][1]. As in the aforementioned paper, training has 2 stages: in first stage only the feature selector is trained (convolutinal layers + last softmax layer), in second stage fully connected layer is added at the end of the network and only the newly added layer is trained.

Stanford background can be trained on two types of data: 1 scale YUV transformed images, 3 scales of laplacian pyramid YUV transformed images. Details are described in the aforementioned work.

## Usage
1. Download *MSRC-21* or *Stanford Background (iccv09)* dataset and unpack them to the *data/* folder.
  * You can update *DATASET_PATH* variable at the beggining of the file acordingly.
3. Run `python generate_msrc.py`, `python generate_iccv_1l.py` or `python generate_iccv_3l.py`.
 * `generate_msrc.py` generates single-scale data for *MSRC* dataset
 * `generate_iccv_1lpy` generates single-scale data for *Stanford Background* dataset
 * `generate_iccv_3l.py` generates 3-scale data for *Stanfrod Background* dataset.
4. Run `python train_2step.py network.conf` for single-scale version of generated data or `python train_2step_3l.py network.conf` for 3-scale version of generated data. `network.conf` file contains a simple network configuration and path to the generated input files. Content of the `network.conf` file will be documented soon.

## Results
3-scale version of network archiveves pixel accuracy of 74.5% on Stanford Background dataset. Farabet-Pami in their paper state 78.8% pixel accuracy. Both results are without oversegmentation.

## Visualization
During runtime, framework generates `output.log` file. Data from it can be visualized using `plot_cost.py` script. Just run
```
    python plot_cost.py output.log
```
Script plots graphs of training cost, validation cost and validation error using *pyplot*.

## Plans
* Usage of *SIFT Flow* dataset
* Framework will be able to load network architecture from configuration file
* Input data generation will be configured through special configuration file
* Support for oversegmentation methods (currently not supported) like superpixels
* Implementation of the Inception layer described in [GoogLeNet paper][2]

#### Master thesis work
**Ivan Borko**, Faculty of electrical engineering and computing, University of Zagreb, Croatia

2014/2015
[1]:http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf
[2]:http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
