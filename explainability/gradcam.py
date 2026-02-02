import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAMExplainer:
    def __init__(self, model, target_layer):
        """
        Wrapper for Grad-CAM.
        Args:
            model (nn.Module): The trained model.
            target_layer (nn.Module): The target convolutional layer (usually the last one).
        """
        self.model = model
        self.target_layer = target_layer
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def explain(self, input_tensor, original_image, target_class=None):
        """
        Generate Grad-CAM heatmap.
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W).
            original_image (np.array): Original image (H, W, 3) normalized [0,1].
            target_class (int, optional): The class index to visualize. If None, highest pred.
        Returns:
            np.array: Visualization image.
        """
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        
        # Generate grayscale CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay on image
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        return visualization
