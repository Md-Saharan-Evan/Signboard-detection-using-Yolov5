import pandas as pd
import os
import shutil

# Define the base directories for images and annotations
base_image_dir = '/Users/mdsaharanevan/Desktop/Yolo_scratch/data/global_images'
base_annotation_dir = '/Users/mdsaharanevan/Desktop/Yolo_scratch/data/global_annotations'
yolo_data_dir = '/Users/mdsaharanevan/Desktop/Yolo_scratch/yolov5/data'

# Load CSV files correctly according to their names
train_df = pd.read_csv('/Users/mdsaharanevan/Desktop/Yolo_scratch/section1-group4/train.csv')  # Correctly labeled as train
val_df = pd.read_csv('/Users/mdsaharanevan/Desktop/Yolo_scratch/section1-group4/val.csv')  # Correctly labeled as val
test_df = pd.read_csv('/Users/mdsaharanevan/Desktop/Yolo_scratch/section1-group4/test.csv')  # Correctly labeled as test

def prepare_dataset(df, type):
    images_dir = os.path.join(yolo_data_dir, f'{type}/images')
    labels_dir = os.path.join(yolo_data_dir, f'{type}/labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for idx, row in df.iterrows():
        src_image_path = os.path.join(base_image_dir, row[0])
        src_label_path = os.path.join(base_annotation_dir, row[1])
        dest_image_path = os.path.join(images_dir, row[0])
        dest_label_path = os.path.join(labels_dir, row[1])

        # Copy files to the appropriate directory
        shutil.copy(src_image_path, dest_image_path)
        shutil.copy(src_label_path, dest_label_path)

# Prepare datasets
prepare_dataset(train_df, 'train')
prepare_dataset(val_df, 'val')
prepare_dataset(test_df, 'test')
