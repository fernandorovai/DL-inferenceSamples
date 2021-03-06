# Converting Faster RCNN to optimized model using OpenVINO #


## Check model node input and outputs 
    python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo/utils/summarize_graph.py --input_model=frozen_inference_graph.pb

    # output example
    1 input(s) detected:
    Name: image_tensor, type: uint8, shape: (-1,-1,-1,3)
    20 output(s) detected:
    BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/switch_t
    BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/switch_f
    BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/switch_t
    BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/switch_f
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/switch_t
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/switch_f
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/switch_t
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/switch_f
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_1/switch_t
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_1/switch_f
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_1/cond/switch_t
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_1/cond/switch_f
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_3/switch_t
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_3/switch_f
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_3/cond/switch_t
    SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_3/cond/switch_f
    detection_boxes
    detection_scores
    num_detections
    detection_classes

## Convert model to IR F32
    python3 /home/junior/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \ 
    --input_model=frozen_inference_graph.pb \
    --tensorflow_use_custom_operations_config=/opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
    --tensorflow_object_detection_api_pipeline_config /pipeline.config \
    --data_type FP32 \ 
    --reverse_input_channels

## Convert model to IR F16
    python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \
    --input_model=frozen_inference_graph.pb \
     --tensorflow_use_custom_operations_config=/opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json \
     --tensorflow_object_detection_api_pipeline_config pipeline.config \
     --data_type FP32 \
     --reverse_input_channels \
     --data_type FP16

    The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the Inference Engine documentation for information about this layer.

## Running inference Script CPU
    python RCNN_inference_OV.py \
    --model frozen_inference_graph_FP32.xml \
    --input images/ \
    -d CPU \
    --cpu_extension=/home/USER/inference_engine_samples/intel64/Release/lib/libcpu_extension.so \
    --output output/ \
    --labels labels.txt
