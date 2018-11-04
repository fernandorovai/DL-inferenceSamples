#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy as np
from time import time
from openvino.inference_engine import IENetwork, IEPlugin
import logging as log
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool
import time
import pprint

name_label_dict = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles"   ,
5:  "Nuclear bodies"   ,
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus"   ,
8:  "Peroxisomes"   ,
9:  "Endosomes"   ,
10:  "Lysosomes"   ,
11:  "Intermediate filaments",   
12:  "Actin filaments"   ,
13:  "Focal adhesion sites",   
14:  "Microtubules"   ,
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle"   ,
18:  "Microtubule organizing center" ,  
19:  "Centrosome"   ,
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions"  , 
23:  "Mitochondria"   ,
24:  "Aggresome"   ,
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Path to a folder with images", required=True,
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)

    return parser


def load_labels(labelsPath):
    labels = []
    with open(labelsPath, "r") as f:
        for line in f.readlines():
            labels.append(line.rstrip())
    return(labels)

def pre_process_image(imagePath):
    n, c, h, w = [1, 3, 299, 299]
    image = Image.open(imagePath)
    processingImg = image.resize((h, w), resample=Image.BILINEAR)
    #image         = image.convert('RGB')
    processingImg = (np.array(processingImg) - 0) / 255.0
    processingImg = processingImg.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    processingImg = processingImg.reshape((n, c, h, w))
    return image, processingImg, imagePath

def pre_process_image_opencv(imagePath):
    n, c, h, w = [1, 3, 299, 299]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = Image.fromarray(image).resize((h, w), resample=Image.BILINEAR)
    image = cv2.resize(image, (w, h))
    image = (np.array(image) - 0) / 255.0
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = image.reshape((n, c, h, w))
    return image, imagePath


def arrangeParallelData(listDir):
    print("Reading images in parallel...")
    poolSize=4
    # load images into mem using multi-processes
    pool = Pool(poolSize)

    totalStartTime = time.time()
    imagesBulk = pool.map(pre_process_image, listDir)
    totalTime = time.time() - totalStartTime
    pool.close()
    pool.join()
    print("Number of elements %s" % str(len(listDir)))
    print("Total Time Reading Files: %s seconds" % str(round(totalTime,4)))
    return imagesBulk

def drawBox(image, fileName, text):
    image = np.array(image)
    weight = 0
    #unormalize
    # image = np.squeeze(image)
    # image = image.transpose((1, 2, 0))

    print(image.shape)
    image = Image.fromarray(image)
    width, height = image.size
    padding = 5
    xMin = 0 + padding
    yMin = 0 + padding
    xMax = xMin + width - 2*padding
    yMax = yMin + height - 2*padding

    color = (0,255,0)
    draw = ImageDraw.Draw(image)
    for i in range(weight):
        draw.rectangle([(xMin-i, yMin-i), (xMax+i, yMax+i)], outline=color)
   
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 13)
    draw.text((10,height-20), text, font=fnt, fill=(255,255,255,255))

    ######## FP32
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 18)
    draw.text((10, 10), "2fps", font=fnt, fill=(255,255,255,255))
    draw.text((10, height-40), "TensorFlow - CPU (FP32)", font=fnt, fill=(255,255,255,255))
    image.save(os.path.join('selectedImages2fpsTF_CPU32',os.path.basename(fileName)))

    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 18)
    # draw.text((10, 10), "16fps", font=fnt, fill=(255,255,255,255))
    # draw.text((10, height-40), "OpenVINO - CPU (FP32)", font=fnt, fill=(255,255,255,255))
    # image.save(os.path.join('selectedImages16fpsOV_CPU32',os.path.basename(fileName)))   

    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 18)
    # draw.text((10, 10), "23fps", font=fnt, fill=(255,255,255,255))
    # draw.text((10, height-40), "OpenVINO - IntelGPU (FP32)", font=fnt, fill=(255,255,255,255))
    # image.save(os.path.join('selectedImages23fpsOV_iGPU32',os.path.basename(fileName))) 

    ######## FP16
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 18)
    # draw.text((10, 10), "36fps", font=fnt, fill=(255,255,255,255))
    # draw.text((10, height-40), "OpenVINO - IntelGPU (FP16)", font=fnt, fill=(255,255,255,255))
    # image.save(os.path.join('selectedImages36fpsOV_iGPU16',os.path.basename(fileName))) 


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    imagesFolder = args.input

    listDir=[]
    for root, dirs, files in os.walk(imagesFolder):
        for file in files:
            listDir.append(os.path.join(root, file))

    #imagesBulk = arrangeParallelData(listDir)
     

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

   
    # # Load network to the plugin
    exec_net = plugin.load(network=net)
    del net
    # Warmup with last image to avoid caching
    #exec_net.infer(inputs={input_blob: imagesBulk[-1][0]})
    exec_net.infer(inputs={input_blob: np.zeros((1, 3, 299, 299))})

    #accContainer = {}
    #for label in labels:
    #    accContainer[label] = {'total':0, 'correct':0, 'acc':0}

    infer_time = []
    #for image, fileName in imagesBulk:
    for fileName in listDir[:40]:
        image, processedImg, imagePath = pre_process_image(fileName)
        #image, imagePath = pre_process_image_opencv(fileName)
        
        #groundTruth = (os.path.dirname(fileName).split("/")[-1]).lower()
        #if groundTruth not in labels:
        #    continue

        t0 = time.time()
        res = exec_net.infer(inputs={input_blob: processedImg})
        infer_time.append((time.time()-t0)*1000)
        log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
        res = res['dense_2/Sigmoid']
        idx = np.argsort(res[0])[-1]
        print(name_label_dict[idx])
        #accContainer[groundTruth]['total']+=1
        
        #if labels[top_ind[0]] == groundTruth:
        #    accContainer[groundTruth]['correct']+=1

        drawBox(image, fileName, name_label_dict[idx])
    del exec_net
    del plugin
    
    #for label in accContainer:
    #    accContainer[label]['acc'] = accContainer[label]['correct'] / accContainer[label]['total']

    #pprint.pprint(accContainer)
    avgElapsed = np.array(infer_time).mean() / 1000.0
    totalTime = round(np.array(infer_time).sum() / 1000.0,4)
    print("Average Elapsed: %s seconds/image" % str(round(avgElapsed,4)))
    print("Total time: %s seconds" %    str(totalTime))

if __name__ == '__main__':
    sys.exit(main() or 0)

