import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract areas from WSI slides based on the provided HistoQC masks")
    parser.add_argument("--slides_path", type=str, help="Path to the WSI slides location", required=True, default="D:/Develop/UBC/Datasets/Synap/Slides")
    parser.add_argument("--output_path", type=str, help="Path to the output location", required=True, default="./output")

    parser.add_argument("--mask_generator_model_path", type=str, help="Path to the segment anything model checkpoint", default="./models/sam_vit_h.pth")

    parser.add_argument("--patch_classifier_model", type=str, help="Path to the cell classifier model", default="./models/tumor_normal.pt")


    return parser.parse_args()

