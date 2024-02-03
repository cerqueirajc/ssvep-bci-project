#! /bin/sh

for i in $(seq 1 35)
do
 curl "http://bci.med.tsinghua.edu.cn/upload/yijun/S1.mat.7z" --output "~/downloads/tsinghua/S1.mat.7z" --create-dirs &
done

wait

for i in $(seq 1 35)
do
 7z x "S$i.mat.7z"
done
