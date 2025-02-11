#!/bin/bash

# ===========================================================
# Install gcc and build essentials
sudo apt update -y && sudo apt upgrade -y 
sudo apt install build-essential -y

# ===========================================================
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions

# Pre-installation: verify GPU, OS, and download toolkit
cd $HOME
lspci | grep -i nvidia
uname -m && cat /etc/*release
wget -O cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# 3.8 Ubuntu Instructions for network installation
cd $HOME
sudo apt-key del 7fa2af80
wget -O cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit
# sudo reboot

# Post-installation actions
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
echo "export PATH=/usr/local/cuda-12.8/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc

# ===========================================================
# https://github.com/tylerfeldman321/EN605.617
# Compile cuda hello world and running it
git clone https://github.com/tylerfeldman321/EN605.617.git
cd $HOME/EN605.617/module0/
nvcc hello-world.cu -o hello -run

# ===========================================================
# https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html
# Ubuntu instructions with network installation
cd $HOME
sudo apt install linux-headers-$(uname -r)
wget -O cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt install nvidia-open -y
# sudo reboot

# ===========================================================
# https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_linux.md
cd $HOME
sudo apt update -y
sudo apt upgrade -y
sudo apt install build-essential -y
sudo apt install cmake -y
sudo apt install opencl-headers ocl-icd-opencl-dev -y

echo '// C standard includes
#include <stdio.h>

// OpenCL includes
#include <CL/cl.h>

int main()
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs( 0, NULL, &numPlatforms );

    if (CL_err == CL_SUCCESS)
        printf("%u platform(s) found\n", numPlatforms);
    else
        printf("clGetPlatformIDs(%i)\n", CL_err);

    return 0;
}' > Main.c
gcc -Wall -Wextra -D CL_TARGET_OPENCL_VERSION=100 Main.c -o HelloOpenCL -lOpenCL
bash HelloOpenCL

# ===========================================================
# https://github.com/tylerfeldman321/EN605.617
# Compile opencl hello world and run it
cd $HOME/EN605.617/module0/
gcc hello_world_cl.c -o hello_world_cl -lOpenCL && ./hello_world_cl 
