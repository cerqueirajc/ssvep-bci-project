tar --exclude="*predict_proba.npy" -czvf /mnt/mystorage/results_delayed.tar.gz /mnt/mystorage/results_delayed
gsutil cp /mnt/mystorage/results_delayed.tar.gz gs://results-bucket-1