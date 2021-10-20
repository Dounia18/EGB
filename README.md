# EGB: Image Quality Assessment based on Ensemble of Gradient Boosting

EGB is a No-Reference Image Quality Assessment Metric.
## Contents

1. [Abstract](#Abstract)
2. [Getting Started](#getting-started)

    2.1. [Dependencies](#dependencies)
   
    2.2. [Usage Demo](#usage-demo)
   
    2.3. [More Information on the Provided Files](#more-information-on-the-provided-files)
   
3. [Performance Benchmark](#performance-benchmark)
4. [Citation](#citation)
5. [Contact](#contact)

 
## Abstract

Multimedia services are constantly trying to deliver better image quality to users. To meet this need, they must have an effective and reliable tool to assess the perceptual image quality. This is particularly true for image restoration (IR) algorithms, where the image quality assessment (IQA) metric plays a key role in the development of these latter. For instance, the recent advances in IR algorithms, which are mainly due to the adoption of generative adversarial network (GAN)-based methods, have clearly shown the need for a reliable IQA metric highly correlated with
human judgment. In this paper, we propose an ensemble of gradient boosting (EGB) metric based on selected features similarity and ensemble learning. First, we analyzed the capability of features extracted by different layers of deep convolutional neural network (CNN) to characterize the perceptual quality distance between the reference and distorted/processed images. We observed that a subset of these layers is more relevant to the IQA task. Accordingly, we exploited these selected layers to compute the features similarity, which are then used as input to a regression network to predict the image quality score. The regression network consists of three gradient boosting regression models that are combined to derive the final quality score. Experiments were performed on the perceptual image processing algorithms (PIPAL) dataset, which has been used in the NTIRE 2021 perceptual image quality assessment challenge.

## Getting Started

#### Dependencies

- python 3.7
- numpy 1.19.5
- Tensorflow 2.4.1
- catboost 0.24.4
- xgboost  0.90
- lightgbm 2.2.3
To use the GPU with tensorflow, these packages should be added:
- CUDA 11.0
- cuDNN 8.0.4

Install the dependencies. 
```sh
pip install numpy==1.19.5
pip install tensorflow==2.4.1
pip install catboost==0.24.4
pip install xgboost==0.90
pip install lightgbm==2.2.3
```

#### Usage Demo
To run the test file 
```sh
python3 test.py path_to_reference_images path_to_distorted_images
```
For more Informations
```sh
$ python3 test.py --help   
$ python3 test.py -h  
```

#### More Information on the Provided Files

1. The directory "models" contains one hdh5 file and three sav files: Pretrained_model.h5, xgboost_model.sav, lightgbm_model.sav and catboost_model.sav.

    - The output of the Pretrained model is a (1,1536) vector containing the necessary features, that will be fed to the three regressors.
    - The output of our Regressors are a (3,1) vector containing the quality estimation using each model, the final quality score is the average of the three scores.

2. The test file contains the main function that is used to generate the output.txt file.

    - Inputs : 
	    - A path to the reference images. The images should be in this format: xxxx.bmp (the xxxx should contains only letters and numbers ).
	    - A path to the distorted images. The images should be in this format xxxx_yyyy.bmp (the yyyy can be any caracter or special caracter or a combinaison of both).
    - Outputs:
	    - A text file with the name output.txt, containing in each line the name of the distorted image and the quality score.
	    
## Performance Benchmark

 Performance comparison on validation set of PIPAL dataset.
 
|Metric| Main Score ↑| SROCC ↑| PLCC ↑|
|------|:-------------:|:--------:|:-------:|
|PSNR| 0.5464| 0.2547| 0.2916
|NQM|0.7621| 0.3457| 0.4163
|UQI| 1.0334| 0.4858| 0.5475
|SSIM| 0.7383| 0.3399| 0.3984
|MS-SSIM| 1.0496| 0.4863| 0.5632
|IFC| 1.2703| 0.5936| 0.6766
|VIF| 0.9570| 0.4334| 0.5235
|VSNR| 0.6962| 0.3212| 0.3750
|RFSIM| 0.5700| 0.2655| 0.3044
|GSM| 0.8869| 0.4181| 0.4688
|SRSIM| 1.2199| 0.5658| 0.6541
|FSIM| 1.0277 |0.4671| 0.5605
|FSIMc| 1.0265| 0.4678| 0.5586
|VSI| 0.9662| 0.4500| 0.5161
|MAD| 1.2340| 0.6077| 0.6262
|NIQE| 0.1661| 0.0643| 0.1017
|MA |0.4039| 0.2005| 0.2034
|PI |0.3352| 0.1690| 0.1662
|LPIPS-Alex| 1.2738| 0.6275| 0.6462
|LPIPS-VGG| 1.2385| 0.5914 |0.6471
|PieAPP| 1.4034 |0.7062| 0.6972
|WaDIQaM|1.3322| 0.6779| 0.6543|
|DISTS| 1.3600| 0.6742| 0.6858|
|SWD| 1.3291| 0.6611 |0.6680|
|EGB (Our)| 1.5511| 0.7758| 0.7752|
	    
Performance comparison on test set of PIPAL dataset.
	    
|Metric| Main Score ↑| SROCC ↑| PLCC ↑|
|------|:-------------:|:--------:|:-------:|
|PSNR| 0.5262| 0.2493| 0.2769
|NQM|0.7598| 0.3644| 0.3953
|UQI|0.8695| 0.4195| 0.4500
|SSIM|0.7549| 0.3613| 0.3935
|MS-SSIM|0.9624| 0.4617| 0.5006
|IFC|1.0400| 0.4851| 0.5548
|VIF|0.8765| 0.3970| 0.4794
|VSNR|0.7789| 0.3682| 0.4107
|RFSIM|0.6321| 0.3037| 0.3284
|GSM|0.8740| 0.4093| 0.4646
|SRSIM|1.2087| 0.5728| 0.6359
|FSIM|1.0747| 0.5038| 0.5709
|FSIMc|1.0783| 0.5057| 0.5726
|VSI|0.9752| 0.4583| 0.5168
|MAD|1.1237| 0.5433| 0.5804
|NIQE|0.1658| 0.0340| 0.1317
|MA | 0.2873| 0.1404| 0.1468
|PI |0.2490| 0.1036| 0.1454
|LPIPS-Alex|1.1368| 0.5658| 0.5710
|LPIPS-VGG|1.2277| 0.5947| 0.6330
|PieAPP|1.2048| 0.6074| 0.5974
|WaDIQaM|1.1012| 0.5532| 0.5480
|DISTS|1.3421| 0.6548| 0.6873
|SWD|1.2584| 0.6242| 0.6341
|EGB (Our)|1.3774| 0.7003| 0.6771

## Citation
We kindly ask you to reference our paper if you find the repo useful to your research:
```
@inproceedings{hammou2021egb,
  title={Egb: Image quality assessment based on ensemble of gradient boosting},
  author={Hammou, Dounia and Fezza, Sid Ahmed and Hamidouche, Wassim},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={541--549},
  year={2021}
}
```

## Contact 
Hammou Dounia , `dhammou@inttic.dz`

Fezza Sid Ahmed , `sfezza@inttic.dz`

