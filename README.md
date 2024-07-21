# TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering

This repo presents a walkthrough of setting up TRIPS and training with own images in AWS environment.


<div style="text-align: center;">Linus Franke, Darius Rückert, Laura Fink, Marc Stamminger</div>



Point-based radiance field rendering has demonstrated impressive results for novel view synthesis, offering a compelling blend of rendering quality and computational efficiency. However, also latest approaches in this domain are not without their shortcomings. 3D Gaussian Splatting [Kerbl and Kopanas et al. 2023] struggles when tasked with rendering highly detailed scenes, due to blurring and cloudy artifacts. On the other hand, ADOP [Rückert et al. 2022] can accommodate crisper images, but the neural reconstruction network decreases performance, it grapples with temporal instability and it is unable to effectively address large gaps in the point cloud.
In this paper, we present TRIPS (Trilinear Point Splatting), an approach that combines ideas from both Gaussian Splatting and ADOP. The fundamental concept behind our novel technique involves rasterizing points into a screen-space image pyramid, with the selection of the pyramid layer determined by the projected point size. This approach allows rendering arbitrarily large points using a single trilinear write. A lightweight neural network is then used to reconstruct a hole-free image including detail beyond splat resolution. Importantly, our render pipeline is entirely differentiable, allowing for automatic optimization of both point sizes and positions.
Our evaluation demonstrate that TRIPS surpasses existing state-of-the-art methods in terms of rendering quality while maintaining a real-time frame rate of 60 frames per second on readily available hardware. This performance extends to challenging scenarios, such as scenes featuring intricate geometry, expansive landscapes, and auto-exposed footage.

[[Project Page]](https://lfranke.github.io/trips/) [[Paper]](https://arxiv.org/abs/2401.06003) [[Youtube]](https://youtu.be/Nw4A1tIcErQ) [[Supplemental Data]](https://zenodo.org/records/10687419)

## Citation

```
@article{franke2024trips,
    title={TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering},
    author={Linus Franke and Darius R{\"u}ckert and Laura Fink and Marc Stamminger},
    year = {2024},
    journal = {Computer Graphics Forum},
    volume = {43},
    number = {2},
    doi = {https://doi.org/10.1111/cgf.15012}
}

```

## Install Requirements

Supported Operating Systems: Ubuntu 22.04, Windows

Nvidia GPU (lowest we tested was an RTX2070)

Supported Compiler: g++-9 (Linux), MSVC (Windows, we used 19.31.31105.0)

Software Requirement: Conda (Anaconda/Miniconda)

## Set Up AWS EC2 Instance
You have to setup an AWS EC2 instance with NVIDIA GPU. The one we tested was g6.xlarge.

For storage, the package (TRIPS, colmap, miniconda) should take less then 20GB. 
If you aim at training your model with your own images, you should also reserve at least 101x of the size of your images for COLMAP dense reconstruction prior to model training, i.e. reserve 101GB for 1.0GB images.

Then, you can connect with your EC2 instance via SSH and Terminal, or other methods provided by AWS.

## Install Instructions Linux

### Install Ubuntu Dependancies
After the connection, run the command below in the EC2 CLI.

```
sudo apt-get update
sudo apt install git build-essential gcc-9 g++-9 
sudo apt-get install unzip
```
For the viewer, also install:
```
sudo apt install xorg-dev
```
(There exists a headless mode without window management meant for training on a cluster, see below)

### Install NVIDIA GPU Driver
Please refer to Step 1 - Step 5 in the [blogpost](https://www.cherryservers.com/blog/install-cuda-ubuntu) for the installation guide. \
OR you can simply run the command below (latest update: 2024 Jul)
```
sudo apt install nvidia-driver-535
nvidia-smi
```

### Install Miniconda
Please refer to the official installation guide [here](https://docs.anaconda.com/miniconda/#quick-command-line-install). \
OR you can simply run the command below
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
source ~/.bashrc
```

### Clone Repo
```shell
git clone git@github.com:lfranke/TRIPS.git
cd TRIPS/
```

### Create Conda Environment
```shell
cd TRIPS
./create_environment.sh
```

### Install Pytorch
```shell
cd TRIPS
./install_pytorch_precompiled.sh
```

### Install CuDNN

Either download the latest version and add it to the conda environment (where CUDA 11.8 was installed, this [article](https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805) is a useful resource) or install via conda:

```shell
conda activate trips
conda install -y -c conda-forge cudnn=8.9.2
```

For our experiments, we used CuDNN 8.9.5, however the conda installed version (8.9.2) should also work fine.

### Compile TRIPS

```shell
cd TRIPS

conda activate trips

export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CC=gcc-9
export CXX=g++-9
export CUDAHOSTCXX=g++-9

unset CUDA_HOME

mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH="./External/libtorch/;${CONDA}" ..

make
```

`make` can take a VERY long time, especially for some CUDA files.

It is VERY likely that you will get a `undefined reference to ...@GLIBCXX_3.4.30' ` error during linking at around 93% of `make` process, most likely your linker fails to resolve the global and conda version of the c++ standard library.

Consider removing the libstdc++ lib from the conda environment before running `cmake` command

```shell
rm $CONDA/lib/libstdc++.so*
```

## Install Instructions Windows

Please refer to [lfranke/TRIPS](https://github.com/lfranke/TRIPS)

## Install Instructions Docker

Thanks to user [abecadel](https://github.com/abecadel) for providing these Docker instructions.

### Install Docker
Make sure to have docker installed with gpu support enables

### Clone Repo
Please follow the `Clone Repo` session in `Install Instructions Linux` above.

### Build docker image
```
docker build -t trips .
```

### Running training
```
docker run -v {data_path}:/data --gpus all -it trips /bin/bash
./train --config configs/train_normalnet.ini
```

### Running viewer (Linux only)
**The command below is yet to be confirmed!** \
First enable X forwarding from docker
```
sudo xhost +local:docker
```
Now you can run the viewer
```
docker run --device /dev/dri/ -v `pwd`/tt_scenes/:/scenes --rm -it --gpus all --net=host --env DISPLAY=$DISPLAY trips viewer --scene_dir /scenes/tnt_scenes/tt_train
```

## Running on pretrained models

Supplemental materials link: [https://zenodo.org/records/10664666](https://zenodo.org/records/10687419)

After a successful compilation, the best way to get started is to run `viewer` on the *tanks and temples* scenes
using our pretrained models.
First, download the scenes (`tt_scenes.zip`) and extract them into `scenes/`.
Now, download the model checkpoints (`tt_checkpoints.zip`) and extract them into `experiments/`. 

Run the command below in the root directory of `TRIPS` repo
```
wget https://zenodo.org/records/10687419/files/tt_scenes.zip
unzip tt_scenes.zip 
mv tnt_scenes/* scenes/
rm tt_scenes.zip

wget https://zenodo.org/records/10687419/files/tt_checkpoints.zip
unzip tt_checkpoints.zip -d experiments/
rm tt_checkpoints.zip
```

Your folder structure should look like this:

```shell
TRIPS/
    build/
        ...
    experiments/
        checkpoint_train
        checkpoint_playground
        ...
    scenes/
        tt_train/
        tt_playground/
        ...
    ...
```

The supplemental data also includes data for the boat (checkpoint and scene combined in one zip), mipnerf360 scenes (in the resolutions used in the paper) and mipnerf360 checkpoints.

## Viewer
**The command below is yet to be confirmed!** \
Your working directory should be the trips root directory.

### Linux

Start the viewer with

```shell
conda activate trips
export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA/lib
./build/bin/viewer --scene_dir scenes/tt_train
```
(note that `tt_train` is the scene name of the Tanks&Temples locomotive scene)

### Windows

```shell
./build/bin/RelWithDebInfo/viewer.exe  --scene_dir scenes/tt_train
```
(depending on the used shell, the full path `C:\....\TRIPS\build\bin\...` may have to be used)

The path is different to the Linux path, the compile configuration is added (RelWithDebInfo)!

(note that `tt_train` is the scene name of the Tanks&Temples locomotive scene)

### Viewer Controls
The most important keyboard shortcuts are:
  * F1: Switch to 3DView
  * F2: Switch to neural view
  * F3: Switch to split view (default)
  * F4: Switch to point rendering view
  * WASD: Move camera
  * Center Mouse + Drag: Rotate around camera center
  * Left Mouse + Drag: Rotate around world center
  * Right click in 3DView: Select camera
  * Q: Move camera to selected camera

<img  width="400"  src="images/adop_viewer.png"> <img width="400"  src="images/adop_viewer_demo.gif">

By default, TRIPS is compiled with a reduced GUI. If you want all GUI buttons present, you can add a `-DMINIMAL_GUI=OFF` to the first cmake call to compile this in.


## Scene Description

TRIPS uses [ADOP](https://github.com/darglein/ADOP)'s scene format. [ADOP](https://github.com/darglein/ADOP) uses a simple, text-based scene description format.

To run on your scenes you have to convert them into this format. If you have created your scene with COLMAP (like us) you can use the colmap2adop converter. More infos on this topic can be found here: [scenes/README.md](scenes/README.md)

OR you can refer to the `Training` session below.

## Training
### Install COLMAP (for your own images only)
Please refer to the official [documentation](https://colmap.github.io/install.html#build-from-source) to install `COLMAP`.

OR you can simply run the command below
```
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

sudo apt-get install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10

git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
## Quick fix for the failure to detect the appropriate CUDA architecture during the cmake process
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=75
ninja
sudo ninja install
```

### Image Reconstruction and Conversion (for your own images only)
Assume your images are already in `scenes` folder, e.g. `~/TRIPS/scenes/{your_scene_name}/images/`

You can simply run the command below in the root directory of `TRIPS` repo for COLMAP reconstruction (sparse + dense) for your images and COLMAP to ADOP conversion conversion prior to model training.
```
## COLMAP reconstruction
bash ./colmap_reconstruction.sh ~/TRIPS/scenes/{your_scene_name}

## COLMAP to ADOP conversion
./colmap2adop.sh  ~/TRIPS/scenes/{your_scene_name}  ~/TRIPS/scenes/{your_scene_name}
```
> Note: COLMAP dense reconstruction will occupy 100x of the size of your images in your storage. However, you can remove the reconstructions (folder `scenes/{your_scene_name}/sparse`, `scenes/{your_scene_name}/dense`) after COLMAP to ADOP conversion procedure.

### Model Training
The pipeline is fitted to your scenes by the `train` executable.
All training parameters are stored in a separate config file, e.g. `train_normalnet.ini`.
The basic syntax is:

Linux:
```shell
conda activate trips
export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA/lib

./build/bin/train --config configs/train_normalnet.ini
```

Windows:
```shell
./build/bin/RelWithDebInfo/train.exe --config configs/train_normalnet.ini
```
(depending on the shell, the full path `C:\....\TRIPS\build\bin\...` may have to be used)

Make again sure that the working directory is the root directory of `TRIPS`
Otherwise, the loss models will not be found.

The training results will be found in folder `experiments/`, e.g. `~/TRIPS/experiments/YYYY-MM-DD_hh-mm-ss_{your_training_name}`

Two configs are given for the two networks used in the paper: `train_normalnet.ini` and `train_sphericalnet.ini`. \
You can replicate and modify to create your own config files for your images. \
You can also override the options in these configs easily via the command line, e.g.

```shell
./build/bin/train --config configs/{config_filename} --TrainParams.scene_names {your_scene_name} --TrainParams.name {your_training_name}

## Example
./build/bin/train --config configs/train_normalnet.ini --TrainParams.scene_names tt_train --TrainParams.name new_name_for_this_training
```
(note that `tt_train` is the scene name of the Tanks&Temples locomotive scene provided by the sample data)

For scenes with extensive environments, consider adding an environment map with:
```shell
./build/bin/train --config configs/train_normalnet.ini \
--PipelineParams.enable_environment_map true
```

If GPU memory is sparse, consider lowering  `batch_size` (standard is 4),  `inner_batch_size` (standard is 4) or `train_crop_size` (standard is 512) in the config files or via the command line, e.g.
```shell
./build/bin/train --config configs/train_normalnet.ini \
--TrainParams.batch_size 1 \
--TrainParams.inner_batch_size 2 \
--TrainParams.train_crop_size 256
```
(however this may impact quality).

By default, every 8th image is removed during training and used as a test image. If you want to change this split, consider overriding which percentage of images should be kept out of training with:

```shell
./build/bin/train --config configs/train_normalnet.ini \
--TrainParams.train_factor 0.1
```
default is 0.125 (so 1/8).

### Live Viewer during Training
**The command below is yet to be confirmed!** \
An experimental live viewer is implemented which shows the fitting process during training in an OpenGL window.
If headless mode is not required (see below) you can add a `-DLIVE_TRAIN_VIEWER=ON` to the first cmake call to compile this version in.

Note: This will have an impact on training speed, as intermediate (full) images will we rendered during training.

## Headless Mode

If you do not want the viewer application, consider calling cmake with an additional `-DHEADLESS=ON`.
This is usually done for training on remote machines.

## Troubleshooting

* The viewer starts with only one view (the model view) and crashes when switching to a different view
    * This usually means, there are no experiments present for the scene. Ensure that you downloaded the checkpoints and extracted them to the `experiments/` folder or train the scene yourself.

* What belongs in the `scenes/` folder and what in the `experiments/` folder?
    * The `scenes/` folder has the output of the colmap2adop processing. This usually includes the `point_cloud.{ply/bin}`, the images and `poses.txt` (see [scenes/README.md](scenes/README.md)).
    * The experiments folder contains checkpoints and is used to create checkpoints during training. These usually include the used config (params.ini) and subfolders with names based on the epoch (i.e. `ep0600` for epoch 600). These subfolders include the `.pth` torch tensor saving files as well as the test output imagses.
