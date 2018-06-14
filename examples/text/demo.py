import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
import time
import math
from nms import nms

# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# model_def = './models/VGGNet/text/text_polygon_orn_inception_wider2_syn_msra_384x384/deploy.prototxt'
# model_weights = './models/VGGNet/text/text_polygon_orn_inception_wider2_syn_msra_384x384/VGG_text_text_polygon_orn_inception_wider2_syn_msra_384x384_iter_10500.caffemodel'
# model_def = './models/coco/deploy_coco.prototxt'
# model_weights = './models/coco/VGG_text_text_polygon_orn_coco_384x384_iter_40000.caffemodel'
model_def = './models/ic15/deploy_ic15.prototxt'
model_weights = './models/ic15/ic15.caffemodel'
net = caffe.Net(model_def,	 # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': (1,3,768,768)})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
image_resize_height = 768
image_resize_width = 768
net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)

image_path = './examples/text/demo.jpg'
visu_dir = './visu_demo/'
score_threshold = 0.2
overlap_threshold = 0.2

image_name=image_path.split('/')[-1]
image=caffe.io.load_image(image_path)
image_height,image_width,channels=image.shape
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image
starttime = time.time()
# Forward pass.
detections = net.forward()['detection_out']
endtime = time.time()
print (endtime - starttime)
# Parse the outputs.
det_conf = detections[0,0,:,2]
det_x1 = detections[0,0,:,7]
det_y1 = detections[0,0,:,8]
det_x2 = detections[0,0,:,9]
det_y2 = detections[0,0,:,10]
det_x3 = detections[0,0,:,11]
det_y3 = detections[0,0,:,12]
det_x4 = detections[0,0,:,13]
det_y4 = detections[0,0,:,14]
# Get detections with confidence higher than 0.6.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= score_threshold]
top_conf = det_conf[top_indices]
top_x1 = det_x1[top_indices]
top_y1 = det_y1[top_indices]
top_x2 = det_x2[top_indices]
top_y2 = det_y2[top_indices]
top_x3 = det_x3[top_indices]
top_y3 = det_y3[top_indices]
top_x4 = det_x4[top_indices]
top_y4 = det_y4[top_indices]

bboxes=[]
for i in xrange(top_conf.shape[0]):
	x1 = int(round(top_x1[i] * image_width))
	y1 = int(round(top_y1[i] * image_height))
	x2 = int(round(top_x2[i] * image_width))
	y2 = int(round(top_y2[i] * image_height))
	x3 = int(round(top_x3[i] * image_width))
	y3 = int(round(top_y3[i] * image_height))
	x4 = int(round(top_x4[i] * image_width))
	y4 = int(round(top_y4[i] * image_height))
	x1 = max(1, min(x1, image_width - 1))
	x2 = max(1, min(x2, image_width - 1))
	x3 = max(1, min(x3, image_width - 1))
	x4 = max(1, min(x4, image_width - 1))
	y1 = max(1, min(y1, image_height - 1))
	y2 = max(1, min(y2, image_height - 1))
	y3 = max(1, min(y3, image_height - 1))
	y4 = max(1, min(y4, image_height - 1))
	score = top_conf[i]
	bbox=[x1,y1,x2,y2,x3,y3,x4,y4,score]
	bboxes.append(bbox)

# apply non-maximum suppression
dt_lines = sorted(bboxes, key=lambda x:-float(x[8]))
nms_flag = nms(dt_lines,overlap_threshold)
results=[]
for k,dt in enumerate(dt_lines):
	if nms_flag[k]:
		if dt not in results:
			results.append(dt)

# visulization
plt.clf()
plt.imshow(image)
currentAxis = plt.gca()
for result in results:
	score = result[-1]
	x1 = result[0]
	y1 = result[1]
	x2 = result[2]
	y2 = result[3]
	x3 = result[4]
	y3 = result[5]
	x4 = result[6]
	y4 = result[7]
	name = '%.2f'%(score)
	print(name)
	rbox = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
	color_rbox='#00FF00'
	currentAxis.add_patch(plt.Polygon(rbox, fill=False, edgecolor=color_rbox, linewidth=2))

plt.axis('off')
plt.savefig(visu_dir+image_name, dpi=300)

print('success')

