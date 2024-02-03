#! /bin/sh

for i in $(seq 1 70)
do
 curl "http://bci.med.tsinghua.edu.cn/upload/yijun/S$i.mat.7z" --output "~/downloads/tsinghua/S$i.mat.7z" --create-dirs &
done

wait

for i in $(seq 1 70)
do
 7z x "S$i.mat.7z"
done
