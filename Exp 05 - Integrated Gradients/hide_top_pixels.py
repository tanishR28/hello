def hide_top_pixels(x, attr, k_percent):
    importance = attr.abs().sum(dim=1)                      # one number per pixel: [1, H, W]
    threshold = torch.quantile(importance.flatten(), 1 - k_percent / 100)
    mask = (importance >= threshold).float().unsqueeze(1)   # 1 = top-k% pixels, shape [1,1,H,W]
    return x * (1 - mask)                                   # set those pixels to 0
