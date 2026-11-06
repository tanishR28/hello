def compute_gradcam(activations, gradients):
    alpha = gradients.mean(dim=(2, 3), keepdim=True)          # 1. average gradient per channel -> weights
    cam = (alpha * activations).sum(dim=1, keepdim=True)      # 2. weighted sum of the feature maps
    cam = F.relu(cam)                                         # 3. keep only positive evidence
    cam = cam.squeeze().detach().cpu().numpy()                # 4. to a 2-D numpy array
    cam = cam - cam.min()                                     # 5. scale to 0..1
    if cam.max() != 0:
        cam = cam / cam.max()
    return cam
