def classifier_fn(batch):
    """LIME gives a batch of images (numpy). Return class probabilities, shape (n, 1000)."""
    tensors = []
    for im in batch:
        im = np.clip(im, 0, 255).astype(np.uint8)
        tensors.append(preprocess(Image.fromarray(im)))   # same preprocessing as normal
    x = torch.stack(tensors)
    return torch.softmax(model(x), dim=1).numpy()
