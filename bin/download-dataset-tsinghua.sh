#! /bin/sh

for i in {1. .35}
do
 curl "http://bci.med.tsinghua.edu.cn/upload/yijun/S$i.mat.7z" --output "~/downloads/tsinghua/S$i.mat.7z"
 7z x "S$i.mat.7z"
 rm "S$i.mat.7z"
done
