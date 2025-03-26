#!/bin/bash
conda create -n oculus python=3.10
conda activate oculus
python3 -m pip install -e .

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
python3 -m pip install opencv-contrib-python open3d pyyaml dgl h5py pyrealsense2 ipdb atomics gradio


# python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git
# mkdir third-party
# cd third-party
# git clone git@github.com:IDEA-Research/GroundingDINO.git
# cd GroundingDINO
# python3 setup.py install
# python3 -m pip install gdown
# gdown 1pc195KITtCCLteHkxD7mr1w-cG8XAvTs  # download DINO+SAM weights
# gdown 1X-Et3-5TdSWuEUfq9gs_Ng-V-hypmLdB
# gdown 1HR3O-rMv9qJoTrG6GNxi-4ZVREQzZ1Rf

python3 -m pip install drake underactuated manipulation==2023.11.8
python3 -m pip install numpy==1.26.4
python3 -m pip install pupil-apriltags
# python3 -m pip install multical # calibration library https://github.com/oliver-batchelor/multical/tree/master
# python3 -m pip install multical[interactive]