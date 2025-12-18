## tag v1.0
### export
yolo11n 的动态batch加end2end导出\
`python export.py --weights weights/yolo11n.pt --imgsz 736 1280 --dynamic_batch --end2end --simplify`
###
yolo11n 的静态batch加end2end导出\
`python export.py --weights weights/yolo11n.pt --imgsz 736 1280 --end2end --simplify`
###
yolo11n 的静态导出无end2end\
`python export.py --weights weights/yolo11n.pt --imgsz 736 1280 --simplify`
###
**除了yolov10其他版本的导出并无差异**
###
yolov10 的动态batch加end2end导出\
`python export.py --weights weights/yolov10s.pt --imgsz 736 1280 --dynamic_batch --simplify --v10 --yaml yolov10s.yaml`
###
**由于yolov10训练完保存的pt文件只有weights,所以导出需要yaml文件，如果这个pt文件包含结构图，那么就不需要yaml文件**
###
yolo seg模型导出\
`python export.py --weights weights/yolo11s-seg.pt --imgsz 736 1280 --dynamic_batch --end2end --simplify --seg`
### torch infer
yolo11n.pt推理\
`python torch_infer.py --weights weights/yolo11n.pt --source data/1.jpg --img_size 736 1280 --half --save`
###
yolov10s.pt推理\
`python torch_infer.py --weights weights/yolov10s.pt --source data/1.jpg --img_size 736 1280 --save --v10 --yaml yolov10s.yaml`
###
**yolov10 的推理如果pt文件包含结构图，那么不需要yaml文件，否则需要。另外需要加上--v10，不支持--half**

### onnx infer
除yolov10外 onnxruntime end2end模型推理(INMSLayer)\
`python ort_infer.py --model weights/yolo11n.onnx --image data/1.jpg --end2end`
###
yolov10 onnxruntime 模型推理\
`python ort_infer.py --model weights/yolov10s.onnx --image data/1.jpg --v10`
###
ultralytics模型 非end2end onnxruntime 推理\
`python ort_infer.py --model weights/yolo11n.onnx --image data/1.jpg --ultralytics`
###
其他非ultralytics模型 非end2end onnxruntime 推理\
`python ort_infer.py --model weights/yolov7-tiny.onnx --image data/1.jpg`

### trt infer
yolo11n.engine efficient_nms end2end模型推理\
`python trt_infer.py --engine /home/jia/yolo11n.engine --image data/1.jpg --output result.jpg --efficient_end2end`
###
yolo11n.engine end2end模型推理\
`python trt_infer.py --engine /home/jia/yolo11n.engine --image data/1.jpg --output result.jpg --end2end`
###
yolo11n.engine 非end2end模型推理\
`python trt_infer.py --engine /home/jia/yolo11n.engine --image data/1.jpg --output result.jpg --ultralytics`
###
yolov10s.engine 模型推理\
`python trt_infer.py --engine /home/jia/yolov10s.engine --image data/1.jpg --output result.jpg --v10`
###
其他非ultralytics efficient_nms end2end模型推理\
`python trt_infer.py --engine /home/jia/yolov7-tiny.engine --image data/1.jpg --output result.jpg --efficient_end2end`
###
其他非ultralytics end2end模型推理\
`python trt_infer.py --engine /home/jia/yolov7-tiny.engine --image data/1.jpg --output result.jpg --end2end`
###
其他非ultralytics非end2end模型推理\
`python trt_infer.py --engine /home/jia/yolov7-tiny.engine --image data/1.jpg --output result.jpg`

### train
单卡yolo11n 训练\
`python train.py --data data/coco128.yaml --model weights/yolo11n.pt --epochs 300 --batch 64 --device 0 --name "yolo11n_coco128" --plots`

单卡yolov10 训练 如果训练的pt文件包含结构图，则和上面yolo11n训练一样，不需要yaml文件，否则需要\
`python train.py --data data/coco128.yaml --weights weights/yolov10s.pt --epochs 300 --batch 64 --device 0 --name yolov10_coco128 --plots --v10 --yaml yolov10s.yaml`

多卡yolo11n 训练\
`torchrun --nproc_per_node 2 --master_port 10001 train.py --data data/coco128.yaml --model "weights/yolo11n.pt" --epochs 300 --batch 128 --device 0,1 --name yolo11n_coco128 --plots`

多卡yolov10 训练 如果训练的pt文件包含结构图，则和上面yolo11n训练一样，不需要yaml文件，否则需要\
`torchrun --nproc_per_node 2 --master_port 10001 train.py --data data/coco128.yaml --weights weights/yolov10s.pt --epochs 300 --batch 128 --device 0,1 --name yolov10_coco128 --plots --v10 --yaml yolov10s.yaml`

### val
yolo11n 验证\
`python val.py --weights weights/yolo11n.pt --data data/coco.yaml --plot`
###
yolov10s 验证 如果训练的pt文件包含结构图，则和上面yolo11n验证一样，不需要yaml文件，否则需要\
`python val.py --weights weights/yolov10s.pt --data data/coco.yaml --plot --v10 --yaml yolov10s.yaml`
### trt val
1. 生成json文件，如果没有的话\
`python utils/yolo2coco.py ----img_dir xxx --label_dir xxx --output xxx --classes xxx`
2. trt val\
`python ./trt_val.py --engine /home/jia/3classes_int8_entropy.engine --img_dir /home/jia/project/test_val/images/val --coco_json 3classes.json --end2end --conf 0.001`


## how to quant
1. 生成npy类型的校准数据集\
`python utils/prepare_calib.py --image_folder xxx --calibration_size xxx --height xxx --width xxx --output_path xxx`
###
2. int4 int8 fp8量化\
`python onnx_quantization.py --onnx_path xxx --quantize_mode xxx --calibration_data xxx --calib_method xxx --output_path xxx`\
这里的int4使用awq_clip量化方法，int8 fp8使用 max或者entropy量化方法,这里的输入onnx模型需要simplify的，opset 19
###
3. trt生成\
int4\
`trtexec --onnx=yolo11s_int4_dy_320.onnx --saveEngine=quant.engine --int4 --int8 --fp16 --minShapes=images:1x3x320x320 --optShapes=images:1x3x320x320 --maxShapes=images:1x3x320x320`\
int8\
`trtexec --onnx=yolo11s_int8_dy_320.onnx --saveEngine=quant.engine --int8 --fp16 --minShapes=images:1x3x320x320 --optShapes=images:1x3x320x320 --maxShapes=images:1x3x320x320`\
fp8\
`trtexec --onnx=yolo11s_fp8_dy_320.onnx --saveEngine=quant.engine --fp8 --fp16 --stronglyTyped --minShapes=images:1x3x320x320 --optShapes=images:1x3x320x320 --maxShapes=images:1x3x320x320`\
fp16\
`trtexec --onnx=yolo11s_dy_320.onnx --saveEngine=quant.engine --fp16 --minShapes=images:1x3x320x320 --optShapes=images:1x3x320x320 --maxShapes=images:1x3x320x320`