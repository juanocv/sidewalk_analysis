import time
from estimation.analysis import *
from utils import *

# Uncomment these if you already have pre-defined your img1 and img2 paths
img1_path = "generic/images/streetview_id3_heading0.jpg"
img2_path = "generic/images/streetview_3dtest_7.jpg"

detectron_model_path = "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"
deeplab_model_path = "utils/models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth" 
# See available models at detectron2/configs - mind it must be panoptic

if __name__ == "__main__":
    # Download images - only if necessary
    
    # api = StreetViewAPI()

    # Method 1: Passing latitude and longitude
    # img1_path = api.download_image(lat=-23.6703243, lon=-46.5637054, fov=90, heading=256) # Replace params
    # img2_path = api.download_image(lat=None, lon=None) # Replace
    # img2 must be 5-20m from img1 and have same fov, pitch and heading
    
    # Method 2: Passing address
    # add = None # Add COMPLETE address as string
    # img1_path = api.download_image(address=None, fov=None, heading=None) # Replace params
    # img2_path = api.download_image(address=add)
    
    # Initialize detectron2's panoptic model - must be panoptic
    predictor, cfg = initialize_model(detectron_model_path)

    # Capture initial time
    start = time.time()

    
    # Uncommented desired estimation method
    # Apply method 1.1 (MiDaS estimation with detectron2's pre-trained panoptic model)
    sidewalk_w1, margin_w1 = estimate_width_m(img1_path,"detectron2",predictor,cfg,"pavement")
    ela = time.time() - start
    print(f"Method: MiDaS estimation\nModel: Detectron2\nDataset: COCO"
        f"\nSidewalk width ~ {sidewalk_w1:.2f} ± {margin_w1:.2f} meters\nELA: {ela:.2f}s")
    '''

    # Apply method 1.2.1 (MiDaS estimation with oneformers' pre-trained panoptic model on ADE20k)
    sidewalk_w2, margin_w2 = estimate_width_m(img1_path,"oneformer",
    oneformer_model_name="shi-labs/oneformer_ade20k_swin_large",
    oneformer_label_name="sidewalk, pavement",
    refine_kwargs=dict(
        # you can override any block’s parameters here:
        bf_kwargs={"smooth_kernel":7, "min_valid_columns":10, "infer_bottom":"interp", "clamp_to":370},
        pl_kwargs={"min_cols":15, "lower_offset":4.0},
        bands=[(0.00,0.30,27),(0.30,0.65,17),(0.65,1.00,9)],
        kernel_height=7,
        close_iter=2,
        max_gap_x=28,
        max_gap_y=12,
        min_keep_area_px=6000
    ),
    apply_refine=False, debug_vis=True)
    # It is used to refine the sidewalk mask by closing and removing small blobs
    # If you want to use the raw mask, set apply_refine=False
    ela = time.time() - start
    print(f"Method: MiDaS estimation\nModel: Oneformer\nDataset: ADE20k"
        f"\nSidewalk width ~ {sidewalk_w2:.2f} ± {margin_w2:.2f} meters\nELA: {ela:.2f}s")

    # Apply method 1.2.2 (MiDaS estimation with oneformers' pre-trained panoptic model on Cityscapes)
    sidewalk_w2, margin_w2 = estimate_width_m(img1_path,"oneformer",
    oneformer_model_name="shi-labs/oneformer_cityscapes_swin_large",
    oneformer_label_name="sidewalk")
    ela = time.time() - start
    print(f"Method: MiDaS estimation\nModel: Oneformer\nDataset: Cityscapes"
         f"\nSidewalk width ~ {sidewalk_w2:.2f} ± {margin_w2:.2f} meters\nELA: {ela:.2f}s")

    # Apply method 2.1 (3D reconstruction with detectron2's pre-trained panoptic model)
    # sidewalk_w3 = analysis.estimate_width_3dr(img1_path, img2_path, "detectron2", predictor, cfg, "pavement")
    # ela = time.time() - start
    # print(f"Method: 3D reconstruction\nModel: Detectron2\nDataset: COCO"
    #      f"\nSidewalk width ~ {sidewalk_w3:.2f} meters\nELA: {ela:.2f}s")
    
    # Apply method 2.2 (3D reconstruction with oneformer's pre-trained panoptic model)
    # sidewalk_w3 = analysis.estimate_width_3dr(img1_path, img2_path, "oneformer", 
    # oneformer_model_name="shi-labs/oneformer_ade20k_swin_large", 
    # oneformer_label_name="sidewalk, pavement")
    # ela = time.time() - start
    # print(f"Method: 3D reconstruction\nModel: Detectron2\nDataset: ADE20k"
    #      f"\nSidewalk width ~ {sidewalk_w3:.2f} meters\nELA: {ela:.2f}s")

    # Apply method 3.1 (MiDaS estimation with OneFormer + DeepLabv3 ensemble (pre-trained))
    # model_name = "deeplabv3plus_mobilenet"
    # deeplab_model = load_deeplab_cityscapes(model_path, model_name, 19, 16, "cuda")
    # print(f"Loaded model type: {type(deeplab_model)}")
    # sidewalk_w3, margin_w3 = estimate_width_m(img1_path,"ensemble", 
    #  ensemble_model1_name="shi-labs/oneformer_ade20k_swin_large",
    #  ensemble_label1_name="sidewalk, pavement",
    #  ensemble_model2_name=deeplab_model,
    #  device='cuda')
    # ela = time.time() - start
    # print(f"Method: MiDaS estimation\nModel: Ensemble\nDataset: ADE20k + Cityscapes"
    #       f"\nSidewalk width ~ {sidewalk_w3:.2f} ± {margin_w3:.2f} meters\nELA: {ela:.2f}s")


    # Apply method 3.2 (MiDaS estimation with OneFormer (ADE20K) + OneFormer (Cityscapes) ensemble (pre-trained))
    sidewalk_w3, margin_w3 = estimate_width_m(img1_path,"ensemble_oneformer_oneformer", 
     ensemble_model1="shi-labs/oneformer_cityscapes_dinat_large",
     ensemble_model2="shi-labs/oneformer_cityscapes_swin_large",
     ensemble_label1="sidewalk",
     ensemble_label2="sidewalk",
     device='cuda')
    ela = time.time() - start
    print(f"Method: MiDaS estimation\nModel: Ensemble OneFormer x Oneformer\nDataset(s): ADE20k x Cityscapes"
          f"\nSidewalk width ~ {sidewalk_w3:.2f} ± {margin_w3:.2f} meters\nELA: {ela:.2f}s")
    '''
