import time
from analysis import *
from utils import *

# Uncomment these if you already have pre-defined your img1 and img2 paths
img1_path = "generic\images\streetview_3dtest_6.jpg"
img2_path = "generic\images\streetview_3dtest_7.jpg"

model_path = "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml" 
# See available models at detectron2/configs - mind it must be panoptic

if __name__ == "__main__":
    # Download images - only if necessary
    '''
    # api = StreetViewAPI()

    # Method 1: Passing latitude and longitude
    # img1_path = api.download_image(lat=-23.6703243, lon=-46.5637054, fov=90, heading=256) # Replace params
    # img2_path = api.download_image(lat=None, lon=None) # Replace
    # img2 must be 5-20m from img1 and have same fov, pitch and heading
    
    # Method 2: Passing address
    # add = None # Add COMPLETE address as string
    # img1_path = api.download_image(address=None, fov=None, heading=None) # Replace params
    # img2_path = api.download_image(address=add)
    '''
    # Initialize detectron2's panoptic model - must be panoptic
    predictor, cfg = initialize_model(model_path)

    # Capture initial time
    start = time.time()

    # Uncommented desired estimation method
    '''
    # Apply method 1.1 (MiDaS estimation with detectron2's pre-trained panoptic model)
    # sidewalk_w1, margin_w1 = analysis.estimate_width_m(img1_path,"detectron2",predictor,cfg,"pavement")
    # ela = time.time() - start
    # print(f"Method: MiDaS estimation\nModel: Detectron2\nDataset: COCO"
    #      f"\nSidewalk width ~ {sidewalk_w1:.2f} ± {margin_w1:.2f} meters\nELA: {ela:.2f}s")

    # Apply method 1.2 (MiDaS estimation with oneformers' pre-trained panoptic model)
    # sidewalk_w2, margin_w2 = analysis.estimate_width_m(img1_path,"oneformer",
    # oneformer_model_name="shi-labs/oneformer_ade20k_swin_large",
    # oneformer_label_name="sidewalk, pavement")
    # ela = time.time() - start
    # print(f"Method: MiDaS estimation\nModel: Oneformer\nDataset: ADE20k"
    #      f"\nSidewalk width ~ {sidewalk_w2:.2f} ± {margin_w2:.2f} meters\nELA: {ela:.2f}s")

    # Apply method 2 (3D reconstruction with detectron2's pre-trained panoptic model)
    # sidewalk_w3 = analysis.estimate_width_3dr(img1_path, img2_path, predictor, cfg, "detectron2", "pavement")
    # ela = time.time() - start
    # print(f"Method: 3D reconstruction\nModel: Detectron2\nDataset: COCO"
    #      f"\nSidewalk width ~ {sidewalk_w3:.2f} meters\nELA: {ela:.2f}s")
    '''
    
