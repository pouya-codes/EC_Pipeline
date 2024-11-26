import glob, os
# OPENSLIDE_PATH = r'D:/Develop/UBC/openslide/bin'
# os.add_dll_directory(OPENSLIDE_PATH)
# import openslide
import cv2
import json, torch
from datetime import datetime
from src.logging_utils import *
from PIL import Image
import numpy as np
from matplotlib.path import Path
from myparser import parse_args
from src.utils import process_mask, process_mask, DictToAttr
from src.generate_mask import MaskGenerator
import pyvips
from tqdm import tqdm
from torchvision import transforms
from src.model import VanillaModel, VarMIL
# from flask import Flask, request, jsonify
import boto3
from aws_config import *
# app = Flask(__name__)
class SlideProcessor:
    def __init__(self, slides_path, output_path, patch_size = 1024, resize_size = 512, stride = 1, batch_size = 32, tumor_classifiers_threshold = 0.9):
        self.slides_path = slides_path
        self.output_path = output_path
        self.extensions = ['*.svs', '*.tiff', "*.tif", '*.ndpi']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.resize_size = resize_size
        self.stride = stride
        self.batch_size = batch_size
        self.tumor_classifiers_threshold = tumor_classifiers_threshold
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def init_mask_generator(self, model_path):
        log_start_action("Mask generator initialization", f"Loading model from {model_path}")
        try:
            self.mask_generator = MaskGenerator(model_path)
            status = "Success"
            details = ""
        except Exception as e:
            status = "Error"
            details = f"Error: {e}"
        log_end_action("Mask generator initialization", status, details)


    def init_VarMIL(self, model_path):
        log_start_action("VarMIL initialization", f"Loading model from {model_path}")
        try:
            state = torch.load(model_path, map_location=self.device)
            model = VarMIL('resnet34', 2)
            state_dict = state['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('model.', '')
                new_state_dict[new_key] = v
            model.eval()
            self.VarMIL = model.to(self.device)
            self.VarMIL.load_state_dict(new_state_dict, strict=True)
            status = "Success"
            details = ""
        except Exception as e:
            status = "Error"
            details = f"Cannot load VarMIL model from {model_path}\nError: {e}"
        log_end_action("VarMIL initialization", status, details)


    def init_patch_classifier(self, model_path):
        log_start_action("Patch classifier initialization", f"Loading model from {model_path}")
        try:
            model = torch.load(model_path, map_location=self.device)
            self.patch_classifier = model.model.to(self.device)
            # Define the transformations
            self.transform = transforms.Compose([
            transforms.Resize((self.resize_size, self.resize_size)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7784, 0.7081, 0.7951],
                                std=[0.1685, 0.2008, 0.1439])
            ])
            status = "Success"
            details = ""
        except Exception as e:
            status = "Error"
            details = f"Cannot load patch classifier model from {model_path}\nError: {e}"
        log_end_action("Patch classifier initialization", status, details)

    def init_representation_generator(self, model_path):
        log_start_action("Representation generator initialization", f"Loading model from {model_path}")
        try:
            state = torch.load(model_path, map_location=self.device)['model']
            model = VanillaModel('resnet34')
            model.eval()
            self.representation_generator = model.to(self.device)
            _, error_keys = self.representation_generator.load_state_dict(state, strict=False)
            for key in error_keys:
                if 'classifier' not in key:
                    raise ValueError('Only classifier should be among unexpected keys!')
            status = "Success"
            details = ""
        except Exception as e:
            status = "Error"
            details = f"Cannot load representation generator model from {model_path}\nError: {e}"
        log_end_action("Representation generator initialization", status, details)

    def open_wsi_slide(self, slide_path):
        log_start_action("Open slide", f"Opening slide {slide_path}")
        slide = None
        try:
            slide = pyvips.Image.new_from_file(slide_path)
            status = "Success"
            detail = ""
        except Exception as e:
            print(f"Cannot open {slide_path}\nError: {e}")
            status = "Error"
            detail = f"Cannot open {slide_path}\nError: {e}"
        log_end_action("Open slide", status, detail)
        return slide
        
    def process_slides(self, save_regions = False):
        log_start_action("Processing slides", f"Processing slides from {self.slides_path}")

        results = {}
        slides = [slide for ext in self.extensions for slide in glob.glob(os.path.join(self.slides_path, ext))]
        for slide_path in slides:
            print(f"Processing {slide_path}")
            log_start_action("Processing slide", f"Processing slide {slide_path}")
            slide = self.open_wsi_slide(slide_path)
            file_name = os.path.basename(slide_path).split('.')[0]
            slide_dimensions = slide.width, slide.height
            regions = None
            os.makedirs(os.path.join(self.output_path, file_name), exist_ok=True)

            mask_path = os.path.join(self.output_path, file_name + "_mask.png")
            if not os.path.exists(mask_path):
                mask, thumb = self.mask_generator.generate_mask(slide_path)
                cv2.imwrite(mask_path, mask)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            regions = process_mask(mask, slide_dimensions)
            slide_representation = []
            for label, areas in regions.items(): 
                for area in tqdm(areas):

                    x, y, width, height, *path = area if len(area) == 5 else area + [None]
                    x = max(x, 0)
                    y = max(y, 0)
                    width = min(width, slide_dimensions[0] - x)
                    height = min(height, slide_dimensions[1] - y)
                    region = slide.crop(x, y, width, height)
                    region = Image.fromarray(region.numpy())
                    1
                    if (save_regions):
                        os.makedirs(os.path.join(self.output_path, file_name, label), exist_ok=True)
                        img_path = os.path.join(self.output_path, file_name, label, f"{x}_{y}_{width}_{height}.png")
                        region_resized = region.resize((int(region.width // 4), int(region.height // 4)))
                        region_resized.save(img_path)

                    reference = [x, y]
                    path = path[0]
                    path.vertices -= np.array(reference)
                    
                    bbox = path.get_extents()
                    x_min_bbox, y_min_bbox, x_max_bbox, y_max_bbox = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                    
                    # Create a grid of points in the patch
                    x_range = np.arange(int(x_min_bbox), int(x_max_bbox), self.patch_size * self.stride)
                    y_range = np.arange(int(y_min_bbox), int(y_max_bbox), self.patch_size * self.stride)
                    grid_x, grid_y = np.meshgrid(x_range, y_range)
                    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

                    # Check if all points in the patch are inside the polygon
                    inside_points = path.contains_points(grid_points)

                    # Collect patches and their coordinates
                    patches = []
                    patch_coords = []

                    for point, inside in zip(grid_points, inside_points):
                        if inside:
                            x, y = point
                            # print(f"Found patch at {x}, {y}")
                            patch = region.crop((x, y, x + self.patch_size, y + self.patch_size)).convert("RGB")
                            transformed_patch = self.transform(patch)
                            patches.append(transformed_patch)
                            patch_coords.append((x, y))

                    # Process patches in batches
                    for i in range(0, len(patches), self.batch_size):
                        batch_patches = patches[i:i + self.batch_size]
                        # batch_coords = patch_coords[i:i + self.batch_size]

                        if batch_patches:
                            batch_patches = torch.stack(batch_patches).cuda()
                            with torch.no_grad():
                                outputs = self.patch_classifier.forward(batch_patches)
                                probs = torch.softmax(outputs, dim=1)
                                pred_probs = probs.cpu().numpy()
                                labels = np.argmax(pred_probs, axis=1)
                                tumor_possitive = (np.max(pred_probs, axis=1) > self.tumor_classifiers_threshold) & (labels == 1)
                                tumor_patches = [patch for patch, is_tumor in zip(batch_patches, tumor_possitive) if is_tumor]
                                if (tumor_patches):
                                    with torch.no_grad():
                                        representation = self.representation_generator(torch.stack(tumor_patches))
                                        slide_representation.extend(representation)



                                # patch.save(os.path.join(self.output_path, file_name, label, f'patch_{x_r + x}_{ y_r + y}.png'))
            bag = torch.stack(slide_representation)
            bag = bag.unsqueeze(0)
            _, output = self.VarMIL.forward(bag)
            probs = torch.softmax(output, dim=1)
            results[file_name] = probs.cpu().detach().numpy()
            log_end_action("Processing slide", "Success", f"Processed slide {slide_path}")
        return results
            


# @app.route('/process', methods=['POST'])
def main():
    bucket_name = INPUT_BUCKET
    log_start_action("Reading s3 bucket", f"Reading from bucket {bucket_name}")

    # s3 = boto3.client('s3')
    # response = s3.list_objects_v2(Bucket=bucket_name)
    # if 'Contents' not in response:
    #     print("No files found in the specified S3 bucket/prefix")
    #     return
    # for obj in response['Contents']:
    #     s3_key = obj['Key']
    #     print(f"Processing {s3_key}")
    #     slide_path = f'/tmp/{s3_key.split("/")[-1]}'
        
    #     s3.download_file(bucket_name, s3_key, slide_path)
    # log_end_action("Reading s3 bucket", "Success", f"Reading from bucket {bucket_name}")


    slides_path = '/tmp'
    # s3.download_file(s3_bucket, s3_key, slide_path)
    
    # args = parse_args()
    args = {
        'slides_path': slides_path,
        'output_path': '/tmp/output',
        'mask_generator_model_path': 'models/sam_vit_h.pth',
        'patch_classifier_model': 'models/tumor_normal.pt',
        'representation_generator_model': 'models/representation.pth',
        'VarMIL_model': 'models/VarMIL.pth'
    }
    args = DictToAttr(args)
    processor = SlideProcessor(args.slides_path, 
                               args.output_path)
    processor.init_mask_generator(args.mask_generator_model_path)
    processor.init_patch_classifier(args.patch_classifier_model)
    processor.init_representation_generator(args.representation_generator_model)
    processor.init_VarMIL(args.VarMIL_model)
    probabilities = processor.process_slides(save_regions=False)
    results = {}
    for slide, result in probabilities.items():
        results[slide] = result.tolist()
    os.remove(slides_path)
    print(results)
    results_json = json.dumps(results)
    s3.put_object(Bucket=bucket_name, Key="results.json", Body=results_json)

    print(f"Results stored in S3 bucket {bucket_name} with key {results.json}")
    # result = {
    #     "status": "success",
    #     "message": "Slide processed successfully",
    #     "details": results
    # }
    # return jsonify(result)

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "Container is running"}), 200



if __name__ == '__main__':
    main()
    # app.run(host='0.0.0.0', port=5000)