# Bottom-Up Human Pose Estimation by Ranking Heatmap-Guided Adaptive Keypoint Estimates

## Introduction
In this work, We present several schemes that are rarely or unthoroughly studied before for improving keypoint detection and grouping (keypoint regression) performance. First, we **exploit the keypoint heatmaps for pixel-wise keypoint regression** instead of separating them for improving keypoint regression. Second, we adopt **a pixel-wise spatial transformer network** to learn adaptive representations for handling the scale and orientation variance to further improve keypoint regression quality. Last, we present **a joint shape and heatvalue scoring scheme** to promote the estimated poses that are more likely to be true poses. Together with the **tradeoff heatmap estimation loss** for balancing the background and keypoint pixels and thus improving heatmap estimation quality, we get the state-of-the-art bottom-up human pose estimation result.
		
## Main Results
### Results on COCO val2017 without multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 30.7M   | 63.7 | 0.678 | 0.868 |  0.740 |  0.620 |  0.764 | 0.723 | 0.898 |  0.776 |  0.656 |  0.820 |
| **pose_hrnet_w48** |  640x640 | 66.8M   | 170.1 | 0.701 | 0.881 |  0.760 |  0.656 |  0.772 | 0.748 | 0.913 |  0.798 |  0.692 |  0.829 |
| **pose_higher_hrnet_w48** |  640x640 | 66.9M   | 179.5 | 0.713 | 0.884 |  0.770 |  0.675 |  0.773 | 0.758 | 0.916 |  0.810 |  0.709 |  0.831 |

### Results on COCO val2017 with multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 30.7M   | 63.7 | 0.707 | 0.880 |  0.769 |  0.661 |  0.777 | 0.758 | 0.919 |  0.812 |  0.702 |  0.838 |
| **pose_hrnet_w48** |  640x640 | 66.8M   | 170.1 | 0.725 | 0.889 |  0.787 |  0.689 |  0.782 | 0.777 | 0.929 |  0.832 |  0.728 |  0.847 |
| **pose_higher_hrnet_w48** |  640x640 | 66.9M   | 179.5 | 0.729 | 0.892 |  0.788 |  0.693 |  0.785 | 0.782 | 0.931 |  0.834 |  0.732 |  0.854 |

### Results on COCO test-dev2017 without multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 30.7M   | 63.7 | 0.666 | 0.878 |  0.728 |  0.611 |  0.745 | 0.714 | 0.908 |  0.770 |  0.646 |  0.808 |
| **pose_hrnet_w48** |  640x640 | 66.8M   | 170.1 | 0.694 | 0.889 |  0.762 |  0.649 |  0.757 | 0.743 | 0.921 |  0.801 |  0.685 |  0.822 |
| **pose_higher_hrnet_w48** |  640x640 | 66.9M   | 179.5 | 0.702 | 0.895 |  0.773 |  0.665 |  0.756 | 0.751 | 0.926 |  0.811 |  0.701 |  0.821 |

### Results on COCO test-dev2017 with multi-scale test
| Backbone | Input size | #Params | GFLOPs | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w32** |  512x512 | 30.7M   | 63.7 | 0.694 | 0.889 |  0.762 |  0.649 |  0.758 | 0.749 | 0.928 |  0.810 |  0.691 |  0.829 |
| **pose_hrnet_w48** |  640x640 | 66.8M   | 170.1 | 0.714 | 0.898 |  0.783 |  0.678 |  0.768 | 0.769 | 0.937 |  0.830 |  0.717 |  0.841 |
| **pose_higher_hrnet_w48** |  640x640 | 66.9M   | 179.5 | 0.718 | 0.902 |  0.787 |  0.683 |  0.768 | 0.774 | 0.941 |  0.835 |  0.724 |  0.843 |

### Results on CrowdPose test without multi-scale test
| Method             |    AP | Ap .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_hrnet_w32** | 64.9  | 84.5  | 69.6   | 72.7   | 65.5   | 56.1   |
| **pose_hrnet_w48** | 66.1  | 84.6  | 71.2   | 73.4   | 66.9   | 57.1   |
| **pose_higher_hrnet_w48** | 66.2 | 84.9 | 71.4 | 73.6 | 67.0 |  57.6   |

### Results on CrowdPose test with multi-scale test
| Method             |    AP | Ap .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| **pose_hrnet_w32** | 67.5  | 86.1  | 72.6   | 75.5   | 68.2   | 58.2   |
| **pose_hrnet_w48** | 68.2  | 85.7  | 73.4   | 75.9   | 69.0   | 58.9   |
| **pose_higher_hrnet_w48** | 68.2 | 86.2 | 73.6 | 75.8 | 69.1 | 59.1   |


### Note:
- Flip test is used.
- GFLOPs is for convolution and linear layers only.


## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA V100 GPU cards for HRNet-w32 and 8 NVIDIA V100 GPU cards for HRNet-w48. Other platforms are not fully tested.

## Quick start
### Installation
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.  
   - **There is a bug in the CrowdPoseAPI, please reverse https://github.com/Jeff-sjtu/CrowdPose/commit/785e70d269a554b2ba29daf137354103221f479e**
5. Build dcn model:
   ```
   python setup.py develop
   ```
6. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── model
   ├── experiments
   ├── lib
   ├── tools 
   ├── log
   ├── output
   ├── README.md
   ├── requirements.txt
   └── setup.py
   ```

7. Download pretrained models and our well-trained models from zoo([OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EgN4JcOE_KNHqG7coNOT_bABZvMWpaJxpy1J-9y1gduGcQ?e=PeeM2K)) and make models directory look like this:
    ```
    ${POSE_ROOT}
    |-- model
    `-- |-- imagenet
        |   |-- hrnet_w32-36af842e.pth
        |   `-- hrnetv2_w48_imagenet_pretrained.pth
        |-- pose_coco
        |   |-- pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth
        |   |-- pose_hrnet_w48_reg_delaysep_bg01_stn_640_adam_lr1e-3_coco_x140.pth
        |   `-- pose_higher_hrnet_w48_reg_delaysep_bg01_0025_stn_640_adam_lr1e-3_coco_x140.pth
        |-- pose_crowdpose
        |   |-- pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.pth
        |   |-- pose_hrnet_w48_reg_delaysep_bg01_stn_640_adam_lr1e-3_crowdpose_x300.pth
        |   `-- pose_higher_hrnet_w48_reg_delaysep_bg01_0025_stn_640_adam_lr1e-3_crowdpose_x300.pth
        `-- rescore
            |-- final_rescore_coco_kpt.pth
            `-- final_rescore_crowd_pose_kpt.pth
    ```
   
### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. 
Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            `-- images
                |-- train2017.zip
                `-- val2017.zip

**For CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training.
Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- |-- crowdpose
        `-- |-- json
            |   |-- crowdpose_train.json
            |   |-- crowdpose_val.json
            |   |-- crowdpose_trainval.json (generated by tools/crowdpose_concat_train_val.py)
            |   `-- crowdpose_test.json
            `-- images.zip

After downloading data, run `python tools/crowdpose_concat_train_val.py` under `${POSE_ROOT}` to create trainval set.

**For learning to score data**, you can generate your train data using your model following this command:
Get the train data using COCO train2017/Crowdpose trainval set.

```
python tools/rescore_data.py \
    --cfg your_config_file(experiments/coco/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.yaml) \
    TEST.MODEL_FILE your_model_file(model/pose_coco/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth) \
    DATASET.TEST train2017 \
    DATASET.DATASET_TEST cocoscore \ 
    DATASET.GET_RESCORE_DATA True \
    RESCORE.USE False 
```
```
python tools/rescore_data.py \
    --cfg your_config_file(experiments/crowdpose/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.yaml) \
    TEST.MODEL_FILE your_model_file(model/pose_crowdpose/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.pth) \
    DATASET.TEST trainval \
    DATASET.DATASET_TEST crowdposescore \ 
    DATASET.GET_RESCORE_DATA True \
    RESCORE.USE False
```

### Note:
- The model trained using data generated by one model can work on other models also.


### Training and Testing

#### Testing on COCO val2017 dataset without multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth
```

#### Testing on COCO val2017 dataset with multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth \ 
    TEST.SCALE_FACTOR 0.5,1,2
```

#### Testing on crowdpose test dataset without multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE models/pose_crowdpose/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.pth
```

#### Testing on crowdpose test dataset with multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE models/pose_crowdpose/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.pth \ 
    TEST.SCALE_FACTOR 0.5,1,2
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.yaml \
```

#### Training on Crowdpose trainval dataset

```
python tools/train.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.yaml \
```

#### Training your rescore model and test it

```
python tools/rescore_train.py --cfg experiments/crowdpose/rescore_crowdpose.yaml 
python tools/rescore_train.py --cfg experiments/coco/rescore_coco.yaml 
```
```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE model/pose_coco/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth \
    RESCORE.MODEL_FILE model/rescore/final_rescore_coco_kpt.pth
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE model/pose_crowdpose/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.pth \
    RESCORE.MODEL_FILE model/rescore/final_rescore_crowd_pose_kpt.pth
```

#### Using inference demo
```
python tools/inference_demo.py --cfg experiments/inference_demo_coco.yaml \
    --videoFile ../multi_people.mp4 \
    --outputDir output \
    --visthre 0.3 \
    TEST.MODEL_FILE model/pose_coco/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth
python tools/inference_demo.py --cfg experiments/inference_demo_crowdpose.yaml \
    --videoFile ../multi_people.mp4 \
    --outputDir output \
    --visthre 0.3 \
    TEST.MODEL_FILE model/pose_crowdpose/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_crowdpose_x300.pth \
```

The above command will create a video under *output* directory and a lot of pose image under *output/pose* directory. 

### Acknowledge
Our code is mainly based on [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). 

We adopted dcn (deformable convolution network) implemented in [MMDetection](https://github.com/open-mmlab/mmdetection).

### Citation

```
@article{SunGMXLZW20,
  title={Bottom-Up Human Pose Estimation by Ranking Heatmap-Guided Adaptive Keypoint Estimates},
  author={Ke Sun, Zigang Geng, Depu Meng, Bin Xiao, Dong Liu, Zhaoxiang Zhang, Jingdong Wang},
  journal={arXiv preprint arXiv:},
  year={2020}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI}
  year={2019}
}
```


