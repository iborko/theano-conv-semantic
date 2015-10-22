# theano-conv-semantic
*!under development!*

Convolutional deep network framework for semantic segmentation, implemented using Theano.

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
1. Download *MSRC-21*, *Stanford Background (iccv09)* or [KITTI][3] dataset (German Ros version) and unpack them to the `data/` folder.
2. Update dataset locations in `generate-iccv.conf` file and in `generate-kitti.conf`. Also, you can change other parameters like validation set percentage, output folder locations. `generate-*.conf` config files are used during the generation of theano-ready dataset.
3. Run `python generate_msrc.py`, `python generate_iccv_1l.py generate-iccv.conf`, `python generate_iccv_3l.py generate-iccv.conf` or `python generate_kitti.py generate-kitti.conf`.
 * `generate_msrc.py` generates single-scale data for *MSRC* dataset
 * `generate_iccv_1l.py` generates single-scale data for *Stanford Background* dataset
 * `generate_iccv_3l.py` generates 3-scale data for *Stanfrod Background* dataset.
 * `generate_kitti.py` generates 3-scale data from *KITTI* dataset.
4. Run `python train_2step_3l.py file.conf` or `python train_kitti.py file.conf` where `file.conf` is configuration file. Inside it you can set theano dataset location (produced by `generate...` script), network parameters, stopping parameters, etc.
5. To calculate results on test set you can run `validate_iccv.py` or `validate_kitti.py` script. Parameters are described at the bottom of the script. Example: `python validate.py network.conf network-12-34.bin test` where `network-12-34.bin` contains best network parameters (automatically generated during training).

## Results

### Results on Stanford background dataset

Method | Accuracy (%) | Class accuracy (%)
:------|-------------:|-------------------:
Convnet (3 scales) | 75.7 | 59.3
Convnet (3 scales) + superpixels | 76.1 | 59.7

### Results on KITTI

Using RGB images + depth component calculated using stereo vision (spsstereo algorithm) and normalized

Method | Accuracy (%) | Class accuracy (%)
:------|-------------:|-------------------:
Convnet (3 scales) | 73.6 | 42.4
Convnet (3 scales) | 75.1 | 43.1

## Visualization
During runtime, framework generates `output.log` file. Data from it can be visualized using `plot_cost.py` script. Just run
```
    python plot_cost.py output.log
```
Script plots graphs of training cost, validation cost and validation error using *pyplot*.

## Plans
* Usage of *SIFT Flow* dataset
* Framework will be able to load network architecture from configuration file (partially done)
* Input data generation will be configured through special configuration file (partially done)
* Support for oversegmentation methods (currently not supported) like superpixels (done)
* Implementation of the Inception layer described in [GoogLeNet paper][2] (partially done, `helpers/layers/...`)

#### Master thesis work
**Ivan Borko**, Faculty of electrical engineering and computing, University of Zagreb, Croatia

2014/2015

[1]:http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf
[2]:http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
[3]:http://refbase.cvc.uab.es/files/rrg2015.pdf
