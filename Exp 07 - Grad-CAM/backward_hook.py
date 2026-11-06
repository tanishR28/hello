def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]               # d score / d A^k     [1, 2048, h, w]
