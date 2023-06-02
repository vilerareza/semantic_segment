import os
# import numpy as np
# from PIL import Image
# import torch
# import torch.distributed as dist
# import pycocotools.mask as maskUtils
# import torch.nn.functional as F
# from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from sam import SegmentAnything
# from coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL


class SemanticSegmenter:
   
    def __init__(self, checkpoint, device = 'cuda') -> None:
        
        #self.semantic_branch_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        #self.semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(0)
        self.sam = SegmentAnything(checkpoint, device)
        #self.mask_branch_model = self.sam.create_predictor(output_mode = 'coco_rle')
        #self.id2label = CONFIG_COCO_ID2LABEL
        
    def oneformer_coco_segmentation(self, image, oneformer_coco_processor, oneformer_coco_model, rank=0):
        inputs = oneformer_coco_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
        outputs = oneformer_coco_model(**inputs)
        predicted_semantic_map = oneformer_coco_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        return predicted_semantic_map


    def get_semantic_segment_anything(self, img):

        anns = {'annotations': self.mask_branch_model.generate(img)}

        class_ids = self.oneformer_coco_segmentation(Image.fromarray(img),
                                                    self.semantic_branch_processor,
                                                    self.semantic_branch_model)

        semantic_mask = class_ids.clone()

        # Sort the segmentation mask based on area
        anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
        
        for ann in anns['annotations']:

            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            # get the class ids of the valid pixels
            propose_classes_ids = class_ids[valid_mask]
            num_class_proposals = len(torch.unique(propose_classes_ids))
            if num_class_proposals == 1:
                semantic_mask[valid_mask] = propose_classes_ids[0]
                continue
            
            top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices

            semantic_mask[valid_mask] = top_1_propose_class_ids

        semantic_class_in_img = torch.unique(semantic_mask)

        # semantic prediction
        semantic_masks = {}
        for i in range(len(semantic_class_in_img)):
            class_id = semantic_class_in_img[i].item()
            class_name = self.id2label['refined_id2label'][str(semantic_class_in_img[i].item())]
            class_mask = (semantic_mask == semantic_class_in_img[i])
            class_mask = class_mask.cpu().numpy().astype(np.uint8)
            semantic_masks[class_id] = {'class_name':class_name, 'class_mask':class_mask} 

        return semantic_masks