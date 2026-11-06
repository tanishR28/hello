def get_baseline(kind, raw_img, x):
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    if kind == "black":                               # pixel value 0, written in normalised units
        return (torch.zeros_like(x) - m) / s
    if kind == "white":                               # pixel value 1
        return (torch.ones_like(x) - m) / s
    if kind == "blur":                                # blurred copy of the same image
        return preprocess(raw_img.filter(ImageFilter.GaussianBlur(15))).unsqueeze(0)
