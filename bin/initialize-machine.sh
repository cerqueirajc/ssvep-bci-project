#! /bin/sh
sudo apt update

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda-installer.sh
bash ~/miniconda-installer.sh

# create and activate new conda env
source ~/.bashrc
conda create -n masters-env python=3.10
conda activate masters-env

# clone project repository and install
git clone https://github.com/cerqueirajc/ssvep-bci-project.git
pip install -e ssvep-bci-project

# donwload tsinghua experiment files
sudo apt install p7zip-full
