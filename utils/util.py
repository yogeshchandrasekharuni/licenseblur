from datetime import datetime
from typing import List, Tuple
import shutil
import os
from sklearn import model_selection
from tqdm import tqdm

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


def train_test_split(
        data_path,
        image_dir_name: str = 'Image',
        label_dir_name: str = 'yolo_format',
        test_size: float = 0.2,
        delete_after_copy: bool = False
        ) -> None:
    """Splits the images into train and test sets randomly"""

    image_names = os.listdir(os.path.join(data_path, image_dir_name))
    train_image_names, test_image_names = model_selection.train_test_split(image_names, test_size=test_size, random_state=42)
    train_image_names = [image_name for image_name in train_image_names if image_name.endswith('.jpg')]
    test_image_names = [image_name for image_name in test_image_names if image_name.endswith('.jpg')]

    image_source = os.path.join(data_path, image_dir_name)
    annts_source = os.path.join(data_path, label_dir_name)
    train_destination = os.path.join(data_path, 'train')
    test_destination = os.path.join(data_path, 'test')

    if not os.path.isdir(train_destination):
        os.mkdir(train_destination)
    if not os.path.isdir(test_destination):
        os.mkdir(test_destination)

    func = os.rename if delete_after_copy else shutil.copy

    for image_name in tqdm(train_image_names):
        func(os.path.join(image_source, image_name), os.path.join(train_destination, image_name))
        func(os.path.join(annts_source, image_name[:-4] + '.txt'), os.path.join(data_path, 'train', image_name[:-4] + '.txt'))

    for image_name in tqdm(test_image_names):
        func(os.path.join(image_source, image_name), os.path.join(test_destination, image_name))
        func(os.path.join(annts_source, image_name[:-4] + '.txt'), os.path.join(data_path, 'test', image_name[:-4] + '.txt'))


def xyxy2xywh(x_topleft: int, y_topleft: int, x_bottomright: int, y_bottomright: int, img_width: int, img_height: int) -> Tuple[float]:
    """Converts top-left and bottom-right vertices to x_centroid, y_centroid, width and height format"""

    x_centroid = (x_topleft + x_bottomright) / 2
    y_centroid = (y_topleft + y_bottomright) / 2

    w = x_bottomright - x_topleft
    h = y_bottomright - y_topleft

    # normalize by image reso
    x_centroid = x_centroid / img_width
    y_centroid = y_centroid / img_height

    w = w / img_width
    h = h / img_height

    return x_centroid, y_centroid, w, h