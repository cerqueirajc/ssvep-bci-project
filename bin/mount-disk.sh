#! /bin/sh

# Use the following command to list and find the correct device
# ls -l /dev/disk/by-id/google-*

sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir /mnt/mystorage
sudo mount -o discard,defaults /dev/sdb /mnt/mystorage
sudo chmod a+w /mnt/mystorage