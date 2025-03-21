import cv2

def read_rgbimg(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"Failed to load {img_path}")
    #img = img[:400-20,:] # resize img to crop out google's logo which may interfere
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 'imread' loads img as BGR so it must be converted to RGB