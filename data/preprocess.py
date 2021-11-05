from genericpath import exists
import os, sys
from tqdm import tqdm
from PIL import Image
from sklearn import model_selection
from pdb import set_trace
import shutil
from natsort import natsorted
from typing import List
from datetime import datetime


def convert_aolp_to_yolo_format(data_path: str, save_path: str) -> None:
    """Convert annotations which are in the format of 
    [x1, y1, x2, y2] of top left and bottom right corners of bbox to
    [x1, y1, w, h] of bbox centroid and width and height
    """
    image_names = os.listdir(os.path.join(data_path, 'Image'))

    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    two_plate_ids = []
    if os.path.isfile(os.path.join(data_path, 'two_plates.txt')):
        with open(os.path.join(data_path, 'two_plates.txt'), 'r') as f:
            two_plate_ids = [line.replace('\n', '') for line in f.readlines()]
    

    class_ = 0
    for image_name in tqdm(image_names):
        if not image_name.endswith('.jpg'):
            continue

        img = Image.open(os.path.join(data_path, 'Image', image_name))
        img_width, img_height = img.size
        img.close()

        def _save(second_plate=False) -> None:
            """
            Subset LE in AOLP has images where 2 plates are annotated
            If YOLO format already exists in save_path, we will append the next annotation to same file
            """
            if second_plate:
                annts = open(os.path.join(data_path, 'groundtruth_localization', image_name[:-4] + '_2.txt'), 'r').readlines()
            else:
                annts = open(os.path.join(data_path, 'groundtruth_localization', image_name[:-4] + '.txt'), 'r').readlines()
            try:
                x_topleft, y_topleft, x_bottomright, y_bottomright = [int(ant.replace('\n', '')) for ant in annts]
            except ValueError:
                # raised when the coordinate is in scientifc format like "1.75e+002"
                x_topleft, y_topleft, x_bottomright, y_bottomright = [int(float(ant.replace('\n', ''))) for ant in annts]

            x_centroid = (x_topleft + x_bottomright) / 2
            y_centroid = (y_topleft + y_bottomright) / 2

            w = x_bottomright - x_topleft
            h = y_bottomright - y_topleft

            # normalize by image reso
            x_centroid = x_centroid / img_width
            y_centroid = y_centroid / img_height

            w = w / img_width
            h = h / img_height

            new_annts = str(class_) + ' ' + str(x_centroid) + ' ' + str(y_centroid) + ' ' + str(w) + ' ' + str(h) + '\n'
            
            _save_path = os.path.join(save_path, image_name[:-4]+'.txt')

            mode = 'w' if not second_plate else 'a'
            with open(_save_path, mode) as f:
                f.write(new_annts)

        _save()
        if image_name[:-4] in two_plate_ids:
            _save(second_plate=True)




def train_test_split(data_path):
    image_names = os.listdir(os.path.join(data_path, 'Image'))
    train_image_names, test_image_names = model_selection.train_test_split(image_names, test_size=0.2, random_state=42)
    train_image_names = [image_name for image_name in train_image_names if image_name.endswith('.jpg')]
    test_image_names = [image_name for image_name in test_image_names if image_name.endswith('.jpg')]

    image_source = os.path.join(data_path, 'Image')
    annts_source = os.path.join(data_path, 'yolo_format')
    train_destination = os.path.join(data_path, 'train')
    test_destination = os.path.join(data_path, 'test')

    if not os.path.isdir(train_destination):
        os.mkdir(train_destination)
    if not os.path.isdir(test_destination):
        os.mkdir(test_destination)

    for image_name in tqdm(train_image_names):
        shutil.copy(os.path.join(image_source, image_name), os.path.join(train_destination, image_name))
        shutil.copy(os.path.join(annts_source, image_name[:-4] + '.txt'), os.path.join(data_path, 'train', image_name[:-4] + '.txt'))

    for image_name in tqdm(test_image_names):
        shutil.copy(os.path.join(image_source, image_name), os.path.join(test_destination, image_name))
        shutil.copy(os.path.join(annts_source, image_name[:-4] + '.txt'), os.path.join(data_path, 'test', image_name[:-4] + '.txt'))


def combine_subsets(path_to_sets: List[str], save_dir: str) -> None:
    """Given a set of lists (Subset_Ax) from AOLP, combines them into a single set"""

    os.makedirs(os.path.join(save_dir, 'Image'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'groundtruth_localization'), exist_ok=True)

    image_id: int = 1 # keeps track of current image id because image ids are repeated in different subsets

    for set in tqdm(path_to_sets):
        image_names = natsorted(os.listdir(os.path.join(set, 'Image')))
        for image_name in tqdm(image_names, leave=False):
            if not image_name.endswith('.jpg'):
                continue
            
            two_plates_ids = []
            if os.path.isfile(os.path.join(set, 'two_plates.txt')):
                with open(os.path.join(set, 'two_plates.txt'), 'r') as f:
                    two_plates_ids = [line.replace('\n', '') for line in f.readlines()]

            shutil.copy(
                src=os.path.join(set, 'Image', image_name),
                dst=os.path.join(save_dir, 'Image', f'{image_id}.jpg')
            )

            shutil.copy(
                src=os.path.join(set, 'groundtruth_localization', image_name[:-4] + '.txt'),
                dst=os.path.join(save_dir, 'groundtruth_localization', f'{image_id}.txt')
            )

            if image_name[:-4] in two_plates_ids:
                shutil.copy(
                    src=os.path.join(set, 'groundtruth_localization', image_name[:-4] + '_2.txt'),
                    dst=os.path.join(save_dir, 'groundtruth_localization', f'{image_id}_2.txt')
                )

            image_id += 1

def construct_yaml(
        skeleton_yaml_path: str,
        train_path: str,
        validation_path: str,
        save_path: str,
        n_classes: int = 1,
        class_names: List[str] = ['license_plate'],
        ) -> None:
    """Constructs a yaml file for training using a skeleton"""

    with open(skeleton_yaml_path, 'r') as f:
        skeleton_yaml = f.read()
    
    skeleton_yaml = skeleton_yaml.replace('$TRAIN_PATH$', train_path)
    skeleton_yaml = skeleton_yaml.replace('$VALIDATION_PATH$', validation_path)
    skeleton_yaml = skeleton_yaml.replace('$N_CLASSES$', str(n_classes))
    skeleton_yaml = skeleton_yaml.replace('$CLASS_NAMES_LIST$', str(class_names))

    meta_data = f"# This yaml file was constructed on {datetime.now()}\n# Path to skeleton -> {skeleton_yaml_path}\n\n\n"
    skeleton_yaml = meta_data + skeleton_yaml

    with open(save_path, 'w') as f:
        f.write(skeleton_yaml)

def main(path_to_sets: str, save_path: str, skeleton_yaml_path: str) -> None:
    """
    Convert annotations which are in the format of 
    [x1, y1, x2, y2] of top left and bottom right corners of bbox to
    [x1, y1, w, h] of bbox centroid and width and height
    """
    print('Combining subsets...', end='', flush=True)
    combine_subsets(path_to_sets, save_dir=os.path.join(save_path, 'combined'))
    print('Done!')

    print('Converting annotations...', end='', flush=True)
    convert_aolp_to_yolo_format(
        data_path=os.path.join(save_path, "combined"),
        save_path=os.path.join(save_path, "combined/yolo_format")
    )
    print('Done!')

    print('Splitting into train and test sets...', end='', flush=True)
    train_test_split(data_path=os.path.join(save_path, "combined"))
    print('Done!')
    
    print('Constructing yaml...', end='', flush=True)
    construct_yaml(
        skeleton_yaml_path=skeleton_yaml_path,
        train_path=os.path.join(save_path, "combined/train"),
        validation_path=os.path.join(save_path, "combined/test"),
        save_path=os.path.join(save_path, "combined/training_meta_data.yaml")
    )
    print('Done!')

    return


if __name__ == '__main__':
    path_to_sets = [
        "/home/yogesh/Documents/licenseblur/data/AOLP/Subset_AC/Subset_AC",
        "/home/yogesh/Documents/licenseblur/data/AOLP/Subset_LE/Subset_LE/Subset_LE",
        "/home/yogesh/Documents/licenseblur/data/AOLP/Subset_RP/Subset_RP/Subset_RP"
        ]

    main(
        path_to_sets=path_to_sets,
        save_path="/home/yogesh/Documents/licenseblur/data/AOLP",
        skeleton_yaml_path="/home/yogesh/Documents/licenseblur/data/training_meta_data_skeleton.yaml"
    )