def grad_cam(image_path):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)
    x = preprocess(image).unsqueeze(0)
 
    model.zero_grad()
    output = model(x)                                         # forward pass (hook saves activations)
    pred = output.argmax(dim=1).item()
    output[0, pred].backward()                                # backward pass (hook saves gradients)
 
    cam = compute_gradcam(activations, gradients)
    cam = cv2.resize(cam, (original.shape[1], original.shape[0]))     # up-size to the image
 
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.55, heat, 0.45, 0)
    return original, heat, overlay, categories[pred]
