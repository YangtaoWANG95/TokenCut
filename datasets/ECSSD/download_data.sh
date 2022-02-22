wget https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip
wget https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip

unzip images.zip 
unzip ground_truth_mask.zip

rm images.zip ground_truth_mask.zip

mv images img
mv ground_truth_mask gt
