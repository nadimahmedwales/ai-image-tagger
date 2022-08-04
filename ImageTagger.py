#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 11:13:03 2022

@author: nadim
"""

#import tkinter as tk
from tkinter import Tk, ttk, Label
import tkinter.filedialog as fd

# Image classification model, approx 330 MB
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Image classification model, approx 330 MB
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher

# Image segmentation model, approx 160 MB
from transformers import DetrFeatureExtractor, DetrForObjectDetection

import torch


from PIL import Image

class ImageTagger:
    def __init__(self, window):
        self.window = window
        self.files = []
        self.add_window_elements()

    def choose_images(self):
        file = fd.askopenfilenames(parent=self.window, title='Choose Images', filetypes=[('Images', '.jpg .JPG')])
        self.files = list(self.window.splitlist(file))
        ttk.Button(self.window, text='Tag Images', command = self.tag_images).grid(column=0, row=4)
    
  
    def add_window_elements(self):
        ttk.Button(self.window, text="Choose Images", command=self.choose_images).grid(column=0, row=1)
        
        label = Label(self.window, text="Choose the Files to tag. Multiple files are allowed")
        label.grid(column=0, row=2, padx=5)

    def tag_images(self):
        vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        deit_feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
        deit_model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224')
        
        # Image Segmentation model
        seg_feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        seg_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


        for file in self.files:
            print(file)
            image =  Image.open(file)
            categories = []
            vit_inputs = vit_feature_extractor(images=image, return_tensors="pt")
            vit_outputs = vit_model(**vit_inputs)
            vit_logits = vit_outputs.logits
            
            # model predicts one of the 1000 ImageNet classes
            vit_predicted_class_idx = vit_logits.argmax(-1).item()
            categories.append(vit_model.config.id2label[vit_predicted_class_idx])
            
            deit_inputs = deit_feature_extractor(images=image, return_tensors="pt")
            deit_outputs = deit_model(**deit_inputs)
            deit_logits = deit_outputs.logits
            deit_predicted_class_idx = deit_logits.argmax(-1).item()
            categories.append(deit_model.config.id2label[deit_predicted_class_idx])


            # Image segmentation model
            
            # I need to reduce the image size, but keep aspect ratio
            # Make larger side to be 640 pixels
            width, height = image.size
            max_length = 640
            if width > height:
                w_ratio = (max_length / width)
                small_height = int(w_ratio * height)
                image = image.resize((max_length, small_height))
            else:
                h_ratio = (max_length / height)
                small_width = int(h_ratio * width)
                image = image.resize((small_width, max_length))
                
            seg_inputs = seg_feature_extractor(images=image, return_tensors="pt")
            seg_outputs = seg_model(**seg_inputs)

            # convert outputs (bounding boxes and class logits) to COCO API

            target_sizes = torch.tensor([image.size[::-1]])
            seg_results = seg_feature_extractor.post_process(seg_outputs, target_sizes=target_sizes)[0]
            
            for score, label, box in zip(seg_results["scores"], seg_results["labels"], seg_results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                # let's keep detections with score > 0.85 and reject the others
                if score > 0.85:
                    categories.append(seg_model.config.id2label[label.item()])
            
            # Remove duplicates from categories
            categories = list(set(categories))
            cat_length = len(categories)
            for n, category in enumerate(categories):
                if n < cat_length-1:
                    print(category, end=', ')
                else:
                    print(category,'\n')
            print('*' * 20)
            print('\n')
            
            finish_label = Label(self.window, text='Finished tagging images.')
            finish_label.grid(column=0, row=6)
            exit_label = Label(self.window, text='Feel free to exit, or select more images')
            exit_label.grid(column=0, row=7)
    
if __name__ == '__main__':
    gui = Tk()
    gui.title('AI Image Tagger')
    gui.geometry('350x240')
    
    # I made it light blue because I find white to be too harsh on the eyes
    gui.config(bg='#d4ffea') # Light blue
    
    # It's best not to let people resize it.
    gui.resizable(False, False)
    
    # Instantiate our tagger
    tagger = ImageTagger(gui)
    # Let's go!
    gui.mainloop()
    