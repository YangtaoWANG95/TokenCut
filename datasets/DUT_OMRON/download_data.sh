wget http://saliencydetection.net/dut-omron/download/DUT-OMRON-image.zip
wget http://saliencydetection.net/dut-omron/download/DUT-OMRON-gt-pixelwise.zip.zip
unzip DUT-OMRON-gt-pixelwise.zip.zip
unzip DUT-OMRON-image.zip
rm DUT-OMRON-gt-pixelwise.zip.zip
rm DUT-OMRON-image.zip
mv DUT-OMRON-image/ img
mv pixelwiseGT-new-PNG/ gt
rm -r __MACOSX/
