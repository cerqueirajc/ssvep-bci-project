#! /bin/sh

for i in $(seq 1 35)
do
 curl "http://bci.med.tsinghua.edu.cn/upload/yijun/S$i.mat.7z" --output /mnt/mystorage/tsinghua_bci_lab/S$i.mat.7z --create-dirs &
done

wait

for i in $(seq 1 35)
do
 7z x "S$i.mat.7z" -o/mnt/mystorage/tsinghua_bci_lab
done
