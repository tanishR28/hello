def slic_segments(im):
    return segmentation.slic(im, n_segments=60, compactness=10, start_label=0, channel_axis=-1)
