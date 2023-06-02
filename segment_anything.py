import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2 as cv
import numpy as np
import time

class SegmentAnything:

    def __init__(self, checkpoint, device='cuda') -> None:

        # Device -> 'cpu' or 'cuda'
        self.device = device
        # Load model object
        print ('Loading model...')
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        self.sam.to(device=device)
        print ('Model loaded...')

    def create_predictor (self, output_mode='binary_mask'):
        # Create predictor from model object
        # Refer to: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
        self.predictor = SamAutomaticMaskGenerator(
            model = self.sam,
            output_mode=output_mode)
        return self.predictor