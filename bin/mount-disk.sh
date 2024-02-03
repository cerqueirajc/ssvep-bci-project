#! /bin/sh

sudo mkdir /mnt/mystorage
sudo mount -o discard,defaults /dev/sdb /mnt/mystorage
sudo chmod a+w /mnt/mystorage