wget http://saliencydetection.net/duts/download/DUTS-TE.zip
unzip DUTS-TE.zip
rm DUTS-TE.zip
mv DUTS-TE/DUTS-TE-Image/ img
mv DUTS-TE/DUTS-TE-Mask/ gt
rm -r DUTS-TE/
