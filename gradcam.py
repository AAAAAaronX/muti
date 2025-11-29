import torch
import numpy as np
import cv2
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image, structural, material_props, target_index=None):
        self.model.eval()
        predictions, _, _ = self.model(image, structural, material_props)

        if target_index is None:
            pred = predictions.mean(dim=1) 
        else:
            pred = predictions[:, target_index]

        self.model.zero_grad()
        pred.sum().backward() 

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(pooled_grads.size(0)):
            self.activations[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8

        return heatmap.cpu().numpy()