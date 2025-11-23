import os
import sys
import cv2
import json
import numpy as np

# EndoVis 2017 的文本描述字典
class2sents = {
    'background': ['background', 'body tissues', 'organs'],
    'instrument': ['instrument', 'medical instrument', 'tool', 'medical tool'],
    'bipolar_forceps': ['bipolar forceps'],
    'prograsp_forceps': ['prograsp forceps'],
    'large_needle_driver': ['large needle driver', 'needle driver'],
    'vessel_sealer': ['vessel sealer'],
    'grasping_retractor': ['grasping retractor'],
    'monopolar_curved_scissors': ['monopolar curved scissors'],
    'other_medical_instruments': ['other instruments', 'ultrasound probe']
}

# 2017 的类别 ID 顺序 (通常 1-7)
# 0: BG, 1: BF, 2: PF, 3: LND, 4: VS, 5: GR, 6: MCS, 7: Other
target_classes = [
    'background', 
    'bipolar_forceps', 
    'prograsp_forceps', 
    'large_needle_driver', 
    'vessel_sealer', 
    'grasping_retractor', 
    'monopolar_curved_scissors', 
    'other_medical_instruments'
]

def get_one_sample(root_dir, image_file, image_path, save_dir, mask, class_name):
    suffix = '.png'
    if '.jpg' in image_file: suffix = '.jpg'
    elif '.bmp' in image_file: suffix = '.bmp' # 适配 BMP
    
    # 保存处理后的 Mask
    mask_path = os.path.join(save_dir, image_file.replace(suffix, '') + '_{}.png'.format(class_name))
    cv2.imwrite(mask_path, mask)
    
    # 生成 JSON 条目
    cris_data = {
        'img_path': image_path.replace(root_dir, ''),
        'mask_path': mask_path.replace(root_dir, ''),
        'num_sents': len(class2sents[class_name]),
        'sents': class2sents[class_name],
    }
    return cris_data

def process(root_dir, cris_data_file):
    cris_data_list = []
    
    image_dir = os.path.join(root_dir, 'images')
    # 根据你的目录结构，生成的 mask 放在这里
    cris_masks_dir = os.path.join(root_dir, 'cris_masks')
    
    if not os.path.exists(cris_masks_dir):
        os.makedirs(cris_masks_dir)
        
    print(f'Processing images from: {image_dir}')
    image_files = os.listdir(image_dir)
    image_files.sort()
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # 路径替换：images -> annotations, 后缀改为 .bmp
        anno_path = image_path.replace('images', 'annotations').replace('.jpg', '.bmp').replace('.png', '.bmp')
        
        if not os.path.exists(anno_path):
            print(f"Warning: Label not found for {image_file}")
            continue

        # 读取 Mask (灰度图，像素值就是类别ID)
        mask = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. 处理二值 Mask (Instrument vs Background)
        # 只要不是 0 (背景)，就是 Instrument
        target_mask = (mask > 0) * 255
        if target_mask.sum() != 0:
            cris_data_list.append(get_one_sample(root_dir, image_file, image_path, cris_masks_dir, target_mask, 'instrument'))
            
        # 2. 处理具体器械类别 (Class 1-7)
        # 跳过 id=0 (background)
        for class_id, class_name in enumerate(target_classes):
            if class_id == 0: continue 
            
            target_mask = (mask == class_id) * 255
            if target_mask.sum() != 0:
                cris_data_list.append(get_one_sample(root_dir, image_file, image_path, cris_masks_dir, target_mask, class_name))

    # 保存 JSON
    with open(os.path.join(root_dir, cris_data_file), 'w') as f:
        json.dump(cris_data_list, f)
    print(f"Done! Saved to {cris_data_file}")

if __name__ == '__main__':
    root_dir = sys.argv[1]
    cris_data_file = sys.argv[2]
    process(root_dir, cris_data_file)