from generic.api_wrapper import StreetViewAPI
from analysis import *

# Uncomment these if you already have pre-defined your img1 and img2 paths
img1_path = "generic\images\streetview_test_3.jpg" 
img2_path = "generic\images\streetview_test_4.jpg"

model_path = "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml" 
# Replace desired model path for a new at detectron2/configs - it must be panoptic

if __name__ == "__main__":
    # Download images - only if necessary

    # Method 1: Passing latitude and longitude
    # api = StreetViewAPI()
    # img1_path = api.download_image(lat=-23.6703243, lon=-46.5637054, fov=90, heading=256) # Replace params
    # img2_path = api.download_image(lat=None, lon=None) # Replace
    # img2 must be 5-20m from img1 and have same fov, pitch and heading
    
    # Method 2: Passing address
    # add = None # Add COMPLETE address as string
    # img1_path = api.download_image(address=None, fov=None, heading=None) # Replace params
    # img2_path = api.download_image(address=add)

    # Initialize panoptic model - must be panoptic
    predictor, cfg = analysis.initialize_model(model_path)

    # Apply method 1 (MiDaS estimation)
    sidewalk_w1, margin_w1 = analysis.estimate_width_m(img1_path, predictor, cfg)
    print(f"Sidewalk width (method 1) ~ {sidewalk_w1:.2f} ± {margin_w1:.2f} meters")

    # Apply method 2 (3D reconstruction)
    sidewalk_w2 = analysis.estimate_width_3dr(img1_path, img2_path, predictor, cfg)
    print(f"Sidewalk width (method 2) ~ {sidewalk_w2:.2f} meters")

    
