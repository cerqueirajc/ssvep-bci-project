#! /bin/sh
sudo apt update

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh
bash /opt/miniconda-installer.sh
source ~/.bashrc
conda create -n masters-env python=3.10
conda activate masters-env

# clone project repository and install
git clone https://github.com/cerqueirajc/ssvep-bci-project.git
cd ssvep-bci-project
pip install -e ssvep-bci-project

# donwload tsinghua experiment files
apt install p7zip-full
