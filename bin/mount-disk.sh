#! /bin/sh

# Use the following command to list and find the correct device
# ls -l /dev/disk/by-id/google-*
sudo mkdir /mnt/mystorage
sudo mount -o discard,defaults /dev/nvme0n2 /mnt/mystorage
sudo chmod a+w /mnt/mystorage