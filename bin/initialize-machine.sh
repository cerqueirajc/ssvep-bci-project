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
pip install -e .

# donwload tsinghua experiment files
apt install p7zip-full
curl http://bci.med.tsinghua.edu.cn/upload/yijun/S1.mat.7z --output ~/downloads/tsinghua/S1.mat.7z
# for i in {1. .5}
# do
#  echo "Hai $i"
# done