import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

def normalize_2d(two_d):
    img_temp = two_d - np.min(two_d)
    return img_temp/np.max(img_temp)

def colormap_2d_flipud(norm_2d):
    #return cv2.applyColorMap(np.uint8(255 * np.flipud(norm_2d)), cv2.COLORMAP_JET)
    return np.flipud(norm_2d)

def visualize_cam(conf, model, x, name, layer='Conv_1'):
    """Grad-CAM visualization."""
    last_conv_layer = model.get_layer(layer) # MobileNetV2 last conv layer
    if len(x.shape) == 3:
        X = x[np.newaxis, ...]
    else:
        X = x
    preds = model.predict(X)
    targ_class = np.argmax(preds[0])

    output = model.output[:, targ_class]
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([X])
    for i in range(int(last_conv_layer.output.shape[3])):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    img = normalize_2d(X[-1, :, :, -1])

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    superimposed = (heatmap + img) / 2

    # img, superimposed, heatmap - all 2d [n_mels, time hop]
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(131)
    ax.set_axis_off()
    ax.imshow(colormap_2d_flipud(heatmap))
    ax = fig.add_subplot(132)
    ax.set_axis_off()
    ax.imshow(colormap_2d_flipud(superimposed))
    ax.set_title('CAM {} - '.format(conf.labels[targ_class]) + name, fontsize=10)
    ax = fig.add_subplot(133)
    ax.set_axis_off()
    ax.imshow(colormap_2d_flipud(img))
    fig.show()