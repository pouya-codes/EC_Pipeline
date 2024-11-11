import glob, os, gc
# OPENSLIDE_PATH = r'D:/Develop/UBC/openslide/bin'
# os.add_dll_directory(OPENSLIDE_PATH)
# import openslide
import cv2
import json, torch
from PIL import Image
from src.process_file import ImageProcessor
import numpy as np
from matplotlib.path import Path
from myparser import parse_args
from src.utils import process_annotation, process_mask, process_mask, process_qupath_dearray
from src.generate_mask import MaskGenerator
import pyvips
from tqdm import tqdm
from torchvision import transforms
from model import VanillaModel, VarMIL
class SlideProcessor:
    def __init__(self, slides_path, output_path, patch_size = 1024, resize_size = 512, stride = 1):
        self.slides_path = slides_path
        self.output_path = output_path
        self.extensions = ['*.svs', '*.tiff', "*.tif", '*.ndpi']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.resize_size = resize_size
        self.stride = stride
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def init_mask_generator(self, model_path):
        self.mask_generator = MaskGenerator(model_path)


    def init_image_processor(self, model_dir, tile_size = 256, post_processing = True, gpu_ids=[]):
        self.image_processor = ImageProcessor(model_dir, tile_size, post_processing, gpu_ids)

    def init_VarMIL(self, model_path):
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
        print("VarMIL model loaded")


    def init_patch_classifier(self, model_path):
        
        model = torch.load(model_path, map_location=self.device)
        self.patch_classifier = model.model.to(self.device)
        # Define the transformations
        self.transform = transforms.Compose([
        transforms.Resize((self.resize_size, self.resize_size)),  # Resize the image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7784, 0.7081, 0.7951],
                             std=[0.1685, 0.2008, 0.1439])
        ])

    def init_representation_generator(self, model_path):
        state = torch.load(model_path, map_location=self.device)['model']
        model = VanillaModel('resnet34')

        model.eval()
        self.representation_generator = model.to(self.device)
        _, error_keys = self.representation_generator.load_state_dict(state, strict=False)
        for key in error_keys:
            if 'classifier' not in key:
                raise ValueError('Only classifier should be among unexpected keys!')

    def open_wsi_slide(self, slide_path):
        try:
            slide = pyvips.Image.new_from_file(slide_path)
            return slide
        except Exception as e:
            print(f"Cannot open {slide_path}\nError: {e}")
            return None
        
    def process_slides(self, save_regions = False):
        slides = [slide for ext in self.extensions for slide in glob.glob(os.path.join(self.slides_path, ext))]
        for slide_path in slides:
            print(f"Processing {slide_path}")
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
                    # print(f"Processing {label} area {x}, {y}, {width}, {height}")
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if x + width > slide_dimensions[0]:
                        width = slide_dimensions[0] - x
                    if y + height > slide_dimensions[1]:
                        height = slide_dimensions[1] - y
                    
                    region = slide.crop(x, y, width, height)
                    region = Image.fromarray(region.numpy())
                    

                    if (False):
                        os.makedirs(os.path.join(self.output_path, file_name, label), exist_ok=True)
                        img_path = os.path.join(self.output_path, file_name, label, f"{x}_{y}_{width}_{height}.png")
                        region_resized = region.resize((int(region.width // 4), int(region.height // 4)))
                        region_resized.save(img_path)

                    reference = [x, y]
                    path = path[0]
                    path.vertices -= np.array(reference)
                    
                    bbox = path.get_extents()
                    x_min_bbox, y_min_bbox, x_max_bbox, y_max_bbox = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                    
                    for x in range(int(x_min_bbox), int(x_max_bbox), self.patch_size * self.stride):
                        for y in range(int(y_min_bbox), int(y_max_bbox), self.patch_size * self.stride):
                            # Create a grid of points in the patch
                            patch_points = [(x + dx, y + dy) for dx in range(self.patch_size) for dy in range(self.patch_size)]
                            # Check if all points in the patch are inside the polygon
                            if np.all(path.contains_points(patch_points)) or True:
                                print(f"Found patch at {x}, {y}")
                                # Extract the patch
                                # os.makedirs(os.path.join(self.output_path, file_name, label), exist_ok=True)
                                patch = region.crop((x, y, x + self.patch_size, y + self.patch_size)).convert("RGB")
                                transormed_patch = self.transform(patch)
                                transormed_patch = transormed_patch.cuda().unsqueeze(0) 
                                with torch.no_grad():
                                    output = self.patch_classifier.forward(transormed_patch)
                                    probs = torch.softmax(output, dim=1)
                                pred_prob = torch.squeeze(probs).cpu().numpy()
                                label = np.argmax(pred_prob)
                                probability = np.max(pred_prob)
                                # tumor patches
                                if label == 1 and probability > 0.9:
                                    with torch.no_grad():
                                        representation = self.representation_generator(transormed_patch)
                                        representation = representation.squeeze()
                                        slide_representation.append(representation)
                                        print(len(slide_representation))
                                        # print(representation.shape)
                                        # exit()



                                # patch.save(os.path.join(self.output_path, file_name, label, f'patch_{x_r + x}_{ y_r + y}.png'))
            bag = torch.stack(slide_representation)
            bag = bag.unsqueeze(0)
            _, output = self.VarMIL.forward(bag)
            probs = torch.softmax(output, dim=1)
            print(probs)



def main():
    args = parse_args()
    processor = SlideProcessor(args.slides_path, 
                               args.output_path)

    processor.init_mask_generator(args.mask_generator_model_path)
    processor.init_patch_classifier(args.patch_classifier_model)
    processor.init_representation_generator('models/model_patch_balanced_acc.pth')
    processor.init_VarMIL('models/model_overall_acc.pth')

        

    processor.process_slides(save_regions=True)

if __name__ == "__main__":
    main()