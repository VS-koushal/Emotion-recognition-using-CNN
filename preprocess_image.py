import cv2
import os
from PIL import Image

def resize_image(input_path, output_path, shape=(224, 224)):
    img = Image.open(input_path).convert('RGB')
    img = img.resize(shape, Image.ANTIALIAS)
    img.save(output_path)

def preprocess_image_dir(input_dir, output_dir, shape=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel_root)
        os.makedirs(out_root, exist_ok=True)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                in_path = os.path.join(root, file)
                out_path = os.path.join(out_root, file)
                resize_image(in_path, out_path, shape=shape)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Resize/crop images to target shape')
    parser.add_argument('--input', required=True, help='Input image directory')
    parser.add_argument('--output', required=True, help='Output image directory')
    parser.add_argument('--shape', type=int, nargs=2, default=[224,224], help='Target image shape')
    args = parser.parse_args()
    preprocess_image_dir(args.input, args.output, tuple(args.shape))
