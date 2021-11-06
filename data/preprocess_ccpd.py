from logging import root
import os
import sys
from tqdm import tqdm
import numpy as np
import shutil
import PIL
from PIL import Image
import traceback
import glob
sys.path.append(os.getcwd())

from utils.util import train_test_split, construct_yaml, xyxy2xywh

def preprocess_ccpd(input_dir: str, output_dir: str, delete_after_copy: bool = False) -> None:
    """
    Preprocesses the Ccpd dataset.

    Args:
        input_dir (str): The path to the input directory.
        output_dir (str): The path to the output directory.
    """
    # Create the output directory if it does not exist.
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

    unique_image_id = 1
    for image_path in tqdm(image_paths):
        img_name = os.path.basename(image_path)

        try:
            img = Image.open(image_path)
        except PIL.UnidentifiedImageError:
            # corrupted image, skip it
            traceback.print_exc()

        if '&' in img_name:
            # these images contain license plates

            # ------------- these lines of code have been stolen from the official Ccpd dataset preprocessing script
            # https://github.com/detectRecog/CCPD/blob/02aaea15137c4d2fe662e57d257c6822356e9304/rpnet/load_data.py#L90
            iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
            top_left, bottom_right = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')] # topleft, bottomright
            # ------------- end stolen code

            x_centroid, y_centroid, width, height = xyxy2xywh(*top_left, *bottom_right, *img.size)

        if delete_after_copy:
            # move original file
            os.rename(
                src=image_path,
                dst=os.path.join(output_dir, 'images', str(unique_image_id) + '.jpg')
            )
        else:
            # keep original file
            shutil.copy(
                src=image_path,
                dst=os.path.join(output_dir, 'images', str(unique_image_id) + '.jpg')
            )

        # Write the ground truth bounding box to the output file.
        with open(os.path.join(output_dir, 'labels', str(unique_image_id) + '.txt'), 'w') as f:
            # class, x_centroid, y_centroid, width, height
            # f.write('0 '+ str(top_left[0]/img.width) + ' ' + str(top_left[1]/img.height) + ' ' + str(bottom_right[0]/img.width) + ' ' + str(bottom_right[1]/img.height))
            if '&' in img_name:
                f.write(f'0 {x_centroid} {y_centroid} {width} {height}')

        unique_image_id += 1


def combine_all_ccpd_dirs(root_path: str, save_path: str) -> None:
    """Combines all images in the CCPD dirs"""

    os.makedirs(save_path, exist_ok=True)
    image_paths = glob.glob(root_path+"/**/*.jpg", recursive=True)
    print(f'Found {len(image_paths)} JPG files.')

    for image_path in tqdm(image_paths):
        shutil.copy(
            src=image_path,
            dst=os.path.join(save_path, os.path.basename(image_path))
        )



# def main(input_dir: str, output_dir: str, yaml_skeleton_path: str, yaml_save_path: str, delete_after_copy: bool = False) -> None:
def main(root_path: str, save_path: str, yaml_skeleton_path: str, yaml_save_path: str, delete_after_copy: bool = False) -> None:
    """
    Main function.

    Args:
        input_dir (str): The path to the input directory.
        output_dir (str): The path to the output directory.
        yaml_skeleton_path (str): The path to the yaml skeleton file.
        yaml_save_path (str): The path to the yaml save file.
    """

    print('Combining...', flush=True)
    combine_all_ccpd_dirs(
        root_path=root_path,
        save_path=save_path
        )
    print('Done!')

    input_dir = save_path
    output_dir = save_path + '_processed'

    print('Preprocessing...', flush=True)
    preprocess_ccpd(input_dir=input_dir, output_dir=output_dir, delete_after_copy=delete_after_copy)
    print('Done!')

    print('Splitting into train and test...', flush=True)
    train_test_split(
        data_path=output_dir,
        image_dir_name='images',
        label_dir_name='labels',
        delete_after_copy=delete_after_copy
    )
    print('Done!')

    # Construct the yaml file
    print('Constructing yaml file...', flush=True)
    construct_yaml(
        skeleton_yaml_path=yaml_skeleton_path,
        train_path=os.path.join(output_dir, 'train'),
        validation_path=os.path.join(output_dir, 'test'),
        save_path=yaml_save_path
    )
    print('Done!')



if __name__ == '__main__':
    # main(
    #     input_dir='/home/yogesh/Documents/licenseblur/data/CCPD2019/ccpd_combined',
    #     output_dir='/home/yogesh/Documents/licenseblur/data/CCPD2019/ccpd_combined_processed',
    #     delete_after_copy=True,
    #     yaml_save_path='/home/yogesh/Documents/licenseblur/data/CCPD2019/ccpd_combined_processed/ccpd_training_metadata.yaml',
    #     yaml_skeleton_path='/home/yogesh/Documents/licenseblur/utils/training_meta_data_skeleton.yaml'
    # )
    main(
        root_path='/home/yogesh/Documents/licenseblur/data/CCPD2019',
        save_path='/home/yogesh/Documents/licenseblur/data/CCPD2019/ccpd_combined',
        delete_after_copy=True,
        yaml_save_path='/home/yogesh/Documents/licenseblur/data/CCPD2019/ccpd_combined_processed/ccpd_training_metadata.yaml',
        yaml_skeleton_path='/home/yogesh/Documents/licenseblur/utils/training_meta_data_skeleton.yaml'
    )
    # combine_all_ccpd_dirs(
    #     root_path='/home/yogesh/Documents/licenseblur/data/CCPD2019',
    #     save_path='/home/yogesh/Documents/licenseblur/data/CCPD2019/ccpd_combined'
    #     )