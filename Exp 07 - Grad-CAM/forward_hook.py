def forward_hook(module, inp, out):
    global activations
    activations = out                     # A^k : feature maps  [1, 2048, h, w]
