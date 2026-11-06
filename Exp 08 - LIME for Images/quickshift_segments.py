def quickshift_segments(im):
    return segmentation.quickshift(im, kernel_size=3, max_dist=6, ratio=0.5, channel_axis=-1)
