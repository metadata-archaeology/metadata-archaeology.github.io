import os
import glob
import natsort

import imageio
import numpy as np
import cv2


def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):
    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[1], top_left[1])
    p3 = (bottom_right[1], bottom_right[0])
    p4 = (top_left[0], bottom_right[0])

    height = abs(bottom_right[0] - top_left[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height/2))

    if thickness < 0:  # big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
        [top_left_main_rect, bottom_right_main_rect], 
        [top_left_rect_left, bottom_right_rect_left], 
        [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

    return src


filenames = glob.glob("figures/mapd_images_small/ImageNet*.png")
filenames = natsort.natsorted(filenames)
print("Retrieved files:", len(filenames), filenames[:5])

images = []
current_frames = {}
last_category = None

for i, filename in enumerate(filenames):
    img = np.array(imageio.imread(filename))
    
    filename, _ = os.path.splitext(filename)
    class_name = filename.split("_")[8]
    probe_category = " ".join(filename.split("_")[9:])
    print(f"{filename} / {class_name} / {probe_category}")
    
    # Add margin at the bottom
    height_margin = 0.1
    new_shape = (img.shape[0] + int(img.shape[0] * height_margin), img.shape[1], img.shape[2])
    pasted_img = np.empty(new_shape, dtype=img.dtype)
    pasted_img.fill(255)
    pasted_img[:img.shape[0], :img.shape[1], :] = img
    
    # Add boxes for text at the bottom
    starting_margin_px = int(img.shape[0] * 0.05)
    loc = (starting_margin_px, new_shape[0] - starting_margin_px)
    # string = f"{class_name.title()} / {probe_category.title()}"
    string = f"{class_name.title()}"
    (text_width, text_height), baseline = cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    
    box_px_margin = text_height // 2
    
    top_left = (starting_margin_px - box_px_margin, new_shape[0] - starting_margin_px - text_height - box_px_margin)
    bottom_right = (starting_margin_px + text_width + box_px_margin, new_shape[0] - starting_margin_px + box_px_margin)
    # pasted_img = rounded_rectangle(pasted_img, top_left, bottom_right, radius=-1, color=(255, 0, 0, 255))
    pasted_img = cv2.rectangle(pasted_img, top_left, bottom_right, color=(255, 0, 0, 255), thickness=-1)
    pasted_img = cv2.putText(pasted_img, string, loc, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 255), 2, cv2.LINE_AA)
    
    # Add probe category on the right
    string = f"{probe_category.title()}"
    (text_width, text_height), baseline = cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    
    top_left = (new_shape[1] - starting_margin_px - box_px_margin - text_width, top_left[1])
    bottom_right = (new_shape[1] - starting_margin_px + box_px_margin, bottom_right[1])
    
    loc = (top_left[0] + box_px_margin, loc[1])
    pasted_img = cv2.rectangle(pasted_img, top_left, bottom_right, color=(0, 0, 0, 255), thickness=-1)
    pasted_img = cv2.putText(pasted_img, string, loc, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 255), 2, cv2.LINE_AA)
    
    # images.append(pasted_img)
    if last_category is not None and last_category != class_name:
        keys = ["typical", "corrupted", "atypical", "random outputs"]
        for k in keys:
            images.append(current_frames[k])
        current_frames = {}
    
    last_category = class_name
    current_frames[probe_category] = pasted_img.copy()

imageio.mimsave('final.gif', images, duration=1)
