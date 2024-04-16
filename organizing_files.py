import os
from ultralytics import YOLO
def organize_for_yolo(input_path, output_path):
    for i in os.listdir(input_path):
        if i.endswith('jpg'):
            os.rename(os.path.join(input_path, i),os.path.join(output_path,'images', 'train', i))
            print(f"File :{os.path.join(input_path, i)} moved to: {os.path.join(output_path,'images', 'train', i)}")
        if i.endswith('txt'):
            os.rename(os.path.join(input_path, i),os.path.join(output_path, 'labels', 'train', i) )
            print(f"File :{os.path.join(input_path, i)} moved to: {os.path.join(output_path,'labels', 'train', i)}")

def train_split(input_path, output_path, split=0.3):
    print(len(os.listdir(input_path)))
    print(len(os.listdir('datasets/data/labels/train')))

model = YOLO("yolov8n.yaml")

results = model.train(data= 'config.yaml', epochs=3)
success = model.export(format="onnx")

#
# train_split('data/images/train/','data')
#
# #organize_for_yolo('data/scenes/val', 'data/')
#
