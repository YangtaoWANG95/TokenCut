mkdir images

wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip ./images
rm train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip ./images
rm val2014.zip

wget http://images.cocodataset.org/zips/test2014.zip
unzip test2014.zip ./images
rm test.2014.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
rm annotations_trainval2014.zip

