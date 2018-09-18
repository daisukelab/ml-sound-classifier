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

def visualize_cam_audio(conf, model, x, name, layer='Conv_1'):
    """Grad-CAM visualization for audio spectrogram."""
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
    ax.set_title('predicted class activation map', fontsize=8)
    ax.imshow(colormap_2d_flipud(heatmap))
    ax = fig.add_subplot(132)
    ax.set_axis_off()
    ax.imshow(colormap_2d_flipud(superimposed))
    ax.set_title('CAM {} - {}\n-- overlay --'.format(conf.labels[targ_class], name), fontsize=9)
    ax = fig.add_subplot(133)
    ax.set_axis_off()
    ax.set_title(conf.what_is_sample, fontsize=8)
    ax.imshow(colormap_2d_flipud(img))
    fig.show()

"""
TBD
def _imshow_friendly(img):
    img_temp = img - np.min(img)
    img_temp = img_temp/np.max(img_temp)
    friendly = np.uint8(255 * img_temp)
    return friendly

def visualize_cam_image(conf, model, model_weight, test_file_index, datapath, 
                        expected_preds, test_time_aug_param={}):
    ""Grad-CAM visualization for image.""
    d.load_test_as_image(datapath)
    d.create_test_generator(test_time_aug_param)
    model.load_weights(model_weight)
    last_conv_layer = model.get_layer('block5_conv3')
    cur_X_test, cur_y_test = next(d.test_gen)
    x = np.array([cur_X_test[test_file_index]])
    preds = model.predict(x)
    targ_class = np.argmax(preds[0])
    result = calc_soft_acc(expected_preds[test_file_index], preds[0])

    output = model.output[:, targ_class]
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(int(last_conv_layer.output.shape[3])):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    img = next(d.test_gen)[0][test_file_index]
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(131)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    ax.set_axis_off()
    ax.set_title('predicted class activation map', fontsize=8)
    ax.matshow(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = ((heatmap*0.5/np.max(heatmap) + img)) / 1.5
    ax = fig.add_subplot(132)
    ax.set_axis_off()
    ax.imshow(_imshow_friendly(superimposed))
    ax.set_title('%s? %s' % (d.labels[targ_class], 'yes' if result == 1 else 'no'), fontsize=9)
    ax = fig.add_subplot(133)
    ax.set_axis_off()
    ax.set_title(conf.what_is_sample, fontsize=8)
    ax.imshow(_imshow_friendly(img))
    fig.show()
"""