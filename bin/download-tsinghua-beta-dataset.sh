#! /bin/sh

curl "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S1-S10.tar.gz" --output "/mnt/mystorage/tsinghua_beta_dataset/S1-S10.tar.gz" --create-dirs
curl "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S11-S20.tar.gz" --output "/mnt/mystorage/tsinghua_beta_dataset/S11-S20.tar.gz" --create-dirs
curl "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S21-S30.tar.gz" --output "/mnt/mystorage/tsinghua_beta_dataset/S21-S30.tar.gz" --create-dirs
curl "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S31-S40.tar.gz" --output "/mnt/mystorage/tsinghua_beta_dataset/S31-S40.tar.gz" --create-dirs
curl "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S41-S50.tar.gz" --output "/mnt/mystorage/tsinghua_beta_dataset/S41-S50.tar.gz" --create-dirs
curl "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S51-S60.tar.gz" --output "/mnt/mystorage/tsinghua_beta_dataset/S51-S60.tar.gz" --create-dirs
curl "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/S61-S70.tar.gz" --output "/mnt/mystorage/tsinghua_beta_dataset/S61-S70.tar.gz" --create-dirs

tar -xvf "/mnt/mystorage/tsinghua_beta_dataset/S1-S10.tar.gz"  -C "/mnt/mystorage/tsinghua_beta_dataset"
tar -xvf "/mnt/mystorage/tsinghua_beta_dataset/S11-S20.tar.gz" -C "/mnt/mystorage/tsinghua_beta_dataset"
tar -xvf "/mnt/mystorage/tsinghua_beta_dataset/S21-S30.tar.gz" -C "/mnt/mystorage/tsinghua_beta_dataset"
tar -xvf "/mnt/mystorage/tsinghua_beta_dataset/S31-S40.tar.gz" -C "/mnt/mystorage/tsinghua_beta_dataset"
tar -xvf "/mnt/mystorage/tsinghua_beta_dataset/S41-S50.tar.gz" -C "/mnt/mystorage/tsinghua_beta_dataset"
tar -xvf "/mnt/mystorage/tsinghua_beta_dataset/S51-S60.tar.gz" -C "/mnt/mystorage/tsinghua_beta_dataset"
tar -xvf "/mnt/mystorage/tsinghua_beta_dataset/S61-S70.tar.gz" -C "/mnt/mystorage/tsinghua_beta_dataset"