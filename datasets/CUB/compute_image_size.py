import os
import cv2
import pandas as pd

def store_cub_image_sizes(root):
    paths = pd.read_csv(
        os.path.join(root, 'CUB_200_2011', 'images.txt'),
        sep=' ', names=['id', 'path'])

    sizes = pd.DataFrame(columns=['id', 'width', 'height'])
    for i in range(len(paths)):
        path = paths.iloc[i].path
        image_path = os.path.join(root, 'CUB_200_2011/images', path)
        image = cv2.imread(image_path)
        sizes.loc[i] = [paths.iloc[i].id, image.shape[1], image.shape[0]]

    save_path = os.path.join(root, 'CUB_200_2011', 'image_sizes.txt')
    sizes.to_csv(path_or_buf=save_path, sep=' ', index=False, header=False)

if __name__ == '__main__':
    root = '.'
    print('Storing image sizes of cub200 dataset')
    store_cub_image_sizes(root)
