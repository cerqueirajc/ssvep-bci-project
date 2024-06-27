#! /bin/sh
sudo apt update

# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init bash
conda init zsh

# create and activate new conda env
source ~/.bashrc
conda create -n masters-env python=3.10
conda activate masters-env

# clone project repository and install
git clone https://github.com/cerqueirajc/ssvep-bci-project.git
pip install -e ssvep-bci-project

# donwload tsinghua experiment files
sudo apt install p7zip-full
