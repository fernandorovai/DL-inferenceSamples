from openvino.inference_engine import IENetwork, IEPlugin
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import logging as log
import numpy as np
import time
import cv2
import sys
import os

def build_argparser():
	parser = ArgumentParser()
	parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", 
						required=True, type=str)
	parser.add_argument("-i", "--input", help="Path to a folder with images", 
						required=True, type=str)
	parser.add_argument("-l", "--cpu_extension", help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.", 
						type=str, default=None)
	parser.add_argument("-d", "--device",
						help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
							 "will look for a suitable plugin for device specified (CPU by default)", 
						default="CPU", type=str)
	parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
	parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
						default=0.5, type=float)
	parser.add_argument("-o", "--output", help="Path to a folder to save inferred images", 
						required=True, type=str)
	parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
	return parser

def load_labels(labelsPath):
	labels = []
	with open(labelsPath, "r") as f:
		for line in f.readlines():
			labels.append(line.rstrip())
	return(labels)

def pre_process_image_opencv(imagePath, shapes):
	n, c, h, w = shapes
	image 		 = cv2.imread(imagePath)
	processedImg = cv2.resize(image, (w, h))
	processedImg = processedImg.transpose((2, 0, 1))  # Change data layout from HWC to CHW
	processedImg = processedImg.reshape((n, c, h, w))
	return image, processedImg, imagePath

def pre_process_image(imagePath, shapes):
	n, c, h, w = shapes
	image = Image.open(imagePath)
	processedImg 		 = np.array(image.resize((w, h), resample=Image.BILINEAR))
	processedImg 		 = processedImg.transpose((2, 0, 1))  # Change data layout from HWC to CHW
	processedImg 		 = processedImg.reshape((n, c, h, w))
	return np.array(image), processedImg, imagePath

def readInputs(imagesFolder):
	listDir=[]
	for root, dirs, files in os.walk(imagesFolder):
		for file in files:
			listDir.append(os.path.join(root, file))
	return listDir

def run_single_inference(fileName, exec_net, input_blob, out_blob, shapes):
	infer_time = []
	# Pre-process first
	image, imagePath = pre_process_image(fileName, shapes)
	t0 = time.time()
	res = exec_net.infer(inputs={input_blob: np.array(image)})
	infer_time.append((time.time()-t0)*1000)
	log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))

def drawBox(image, fileName, text):
    weight = 5
    #unormalize
    image = np.squeeze(image)
    image = image.transpose((1, 2, 0))

    print(image.shape)
    image = Image.fromarray((image*255).astype('uint8'))
    width, height = image.size
    padding = 5
    xMin = 0 + padding
    yMin = 0 + padding
    xMax = xMin + width - 2*padding
    yMax = yMin + height - 2*padding

    color = (255,0,0)
    draw = ImageDraw.Draw(image)
    for i in range(weight):
        draw.rectangle([(xMin-i, yMin-i), (xMax+i, yMax+i)], outline=color)

    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    draw.text((10,10), text, font=fnt, fill=(255,255,255,255))
    image.save(os.path.basename(fileName))    

def main():
	log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
	args = build_argparser().parse_args()
	model_xml = args.model
	model_bin = os.path.splitext(model_xml)[0] + ".bin"
	if args.labels:
		labels = load_labels(args.labels)
	imagesFolder = args.input

	# Main sync point:
	# in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
	# in the regular mode we start the CURRENT request and immediately wait for it's completion
	is_async_mode = False

	# Get images from input folder
	listDir = readInputs(imagesFolder)

	# Read IR
	log.info("Reading IR...")
	net = IENetwork.from_ir(model=model_xml, weights=model_bin)

	# Plugin initialization for specified device and load extensions library if specified
	log.info("Initializing plugin for {} device...".format(args.device))
	plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)

	if args.cpu_extension and 'CPU' in args.device:
		plugin.add_cpu_extension(args.cpu_extension)
		
	assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
	assert len(net.outputs) == 1, "Sample supports only single output topologies"
	input_blob = next(iter(net.inputs))
	out_blob = next(iter(net.outputs))
	shapes = net.inputs[input_blob]

	# Load network to the plugin
	exec_net = plugin.load(network=net)
	del net

	cur_request_id  = 0
	next_request_id = 1

	for fileName in listDir:
		image, processedImg, imagePath = pre_process_image(fileName, shapes)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		initial_w = image.shape[1]
		initial_h = image.shape[0]
		inf_start = time.time()
		if is_async_mode:
			exec_net.start_async(request_id=next_request_id, inputs={input_blob: processedImg})
		else:
			exec_net.start_async(request_id=cur_request_id, inputs={input_blob: processedImg})
		if exec_net.requests[cur_request_id].wait(-1) == 0:
			inf_end = time.time()
			det_time = inf_end - inf_start

		res = exec_net.requests[cur_request_id].outputs[out_blob]
		for obj in res[0][0]:
			# Draw only objects when probability more than specified threshold
			if obj[2] > args.prob_threshold:
				xmin = int(obj[3] * initial_w)
				ymin = int(obj[4] * initial_h)
				xmax = int(obj[5] * initial_w)
				ymax = int(obj[6] * initial_h)
				class_id = int(obj[1]) -1

				# Draw box and label\class_id
				color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
				det_label = labels[class_id] if labels else str(class_id)
				cv2.putText(image, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
							cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)	
		cv2.imwrite(os.path.join(args.output, os.path.basename(fileName)), image)
		drawBox()
		# Draw performance stats
		inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
			"Inference time: {:.3f} ms".format(det_time * 1000)
		# render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
		async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
			"Async mode is off. Processing request {}".format(cur_request_id)

		print(inf_time_message)
		# cv2.putText(image, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
		# # cv2.putText(image, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
		# cv2.putText(image, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
		# 			(10, 10, 200), 1)

		if is_async_mode:
			cur_request_id, next_request_id = next_request_id, cur_request_id

	del exec_net
	del plugin

main()