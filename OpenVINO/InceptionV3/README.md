# Converting InceptionV3 to optimized model using OpenVINO ######
Check the medium article on how to speed-up your inference time up to 19x using Intel Core processor  
https://medium.com/@fernandorodriguesjunior/speed-up-inceptionv3-inference-time-up-to-18x-using-intel-core-processor-43d742d9bac

<table>
    <tr>
        <td width=300><img src="https://github.com/fernandorovai/DL-inferenceSamples/tree/master/OpenVINO/InceptionV3/docs/selectedImages2fpsTF_CPU32.gif"/></td>
        <td width=300><img src="https://github.com/fernandorovai/DL-inferenceSamples/tree/master/OpenVINO/InceptionV3/docs/selectedImages16fpsOV_CPU32.gif"/></td>
        <td width=300><img src="https://github.com/fernandorovai/DL-inferenceSamples/tree/master/OpenVINO/InceptionV3/docs/selectedImages23fpsOV_iGPU32.gif"/></td>
        <td width=400><img src="https://github.com/fernandorovai/DL-inferenceSamples/tree/master/OpenVINO/InceptionV3/docs/selectedImages36fpsOV_iGPU16.gif"/></td>
    </tr>
    <tr>
        <td colspan=2>
            <ul>
                <li>Transform h5 model into ckpt</li>
                <li>Find model node output (KERAS)</li>
                <li>Check model node output (summary if not in Keras)</li>
                <li>Frozen model</li>
                <li>Convert model to IR </li>
                <li>Convert model to IR </li>
                <li>Running inference </li>
            </ul>
        </td>
    </tr>
</table>

## Transform h5 model into ckpt
    sess = tf.keras.backend.get_session()
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/home/junior/Documents/Fun/ProteinClassifier/full_protein.ckpt")

## Find model node output (KERAS)
    print(model.output.op.name)

## Check model node output (summary if not in Keras)
    python /home/junior/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo/utils/summarize_graph.py --input_model=inception_v1_inf_graph.pb


## Frozen model
    python /home/junior/anaconda3/envs/tf110_py37_conda/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py \
    --input_meta_graph=modeĺ.ckpt.meta \
    --input_checkpoint=modeĺ.ckpt \
    --output_graph=keras_frozen.pb \
    --output_node_names="dense_2/Sigmoid" \
    --input_binary=true


## Convert model to IR 
    python3 /home/junior/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py --input_model /home/junior/Documents/Fun/ProteinClassifier/keras_frozen.pb --input_shape [1,299,299,3]

## Running inference
    python inceptionV3_Inference_OV.py -m keras_frozen.xml -i testSmallSetRGB/ -d GPU

## Benchmarks
<table>
<tr>
<td colspan="3"><img src="https://github.com/fernandorovai/DL-inferenceSamples/tree/master/OpenVINO/InceptionV3/docs/InceptionV3-Inference-Core i7-7500U.png"/></td>
</tr>
</table>

### TensorFlow + keras FP32
    [100/100]
    Average Elapsed: 0.4292 seconds/image
    Total time: 42.92369556427002 seconds
    2.33 images/sec

### OpenVINO iGPU FP32
    [100/100] 
    Average Elapsed: 0.0426 seconds/image
    Total time: 4.2619 seconds
    23.47 images/sec

### OpenVINO iGPU FP16
    [100/100] 
    Average Elapsed: 0.0282 seconds/image
    Total time: 5.6365 seconds
    35.46 images/sec

### OpenVINO CPU FP32
    [100/100] 
    Average Elapsed: 0.0632 seconds/image
    Total time: 12.6409 seconds
    15.82 images/sec