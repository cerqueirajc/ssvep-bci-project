#! /bin/sh

sudo mkdir /mnt/mystorage
sudo mount -o discard,defaults /dev/nvme0n2 /mnt/mystorage
sudo chmod a+w /mnt/mystorage