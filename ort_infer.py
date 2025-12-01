import cv2
import numpy as np
import onnxruntime as ort
import argparse


class YOLO_ONNX_Runner:
    def __init__(self, model_path, confidence_thres=0.4, iou_thres=0.7):
        self.conf_thres = confidence_thres
        self.iou_thres = iou_thres

        # 优先使用 CUDA, 其次 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"模型加载成功，使用设备: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)
        
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        print(f"模型输入节点: {self.input_name}, 形状: {self.input_shape}")

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_name = model_outputs[0].name
        self.output_shape = model_outputs[0].shape
        print(f"模型输出节点: {self.output_name}, 形状: {self.output_shape}")

    def preprocess(self, image_src):
        self.img_h, self.img_w = image_src.shape[:2]
        # 1. Letterbox Resize (保持长宽比，填充灰色)
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]
        scale = min(self.input_height / self.img_h, self.input_width / self.img_w)
        new_h, new_w = int(self.img_h * scale), int(self.img_w * scale)
        
        image_resized = cv2.resize(image_src, (new_w, new_h))

        # 创建画布并填充
        image_padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        # 计算居中偏移量
        dw = (self.input_width - new_w) // 2
        dh = (self.input_height - new_h) // 2
        image_padded[dh:dh+new_h, dw:dw+new_w, :] = image_resized
        
        # 2. 归一化 & 转换
        image_data = image_padded.transpose(2, 0, 1) # HWC -> CHW
        image_data = np.expand_dims(image_data, axis=0) # Add Batch Dim
        image_data = image_data.astype(np.float32) / 255.0 # 0-255 -> 0.0-1.0
        return image_data, scale, (dw, dh)

    def postprocess(self, output, scale, pad, v8):
        """
        后处理：解析 YOLO 输出, NMS, 坐标还原
        YOLOv8 输出形状通常为: [1, 4 + num_classes, num_anchors]
        例如: [1, 84, 8400] -> 4个坐标 + 80个类别
        """
        if v8:
            # 1. Transpose: [1, 84, 8400] -> [1, 8400, 84]
            output = np.transpose(output, (0, 2, 1))
        
        # 去掉 Batch 维度 -> [8400, 84]
        prediction = output[0]
        
        # 2. 拆分 Box 和 Scores
        # cx, cy, w, h
        boxes = prediction[:, 0:4]
        if v8:
            # classes scores
            scores = prediction[:, 4:]
        else:
            scores = prediction[:, 4:5] * prediction[:, 5:]
        
        # 获取最大置信度的类别和分数
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        
        # 3. 初步过滤 (Confidence Threshold)
        mask = max_scores >= self.conf_thres
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        max_scores = max_scores[mask]
        
        if len(boxes) == 0:
            return [], [], []

        # 4. 坐标转换: cx,cy,w,h -> x1,y1,x2,y2 (用于 NMS)
        # 这里的 boxes 还是基于 640x640 (input_size) 的
        nms_boxes = np.copy(boxes)
        nms_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        nms_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        nms_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        nms_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # 5. NMS (Non-Maximum Suppression)
        # cv2.dnn.NMSBoxes 需要 (x, y, w, h) 格式，或者我们可以用 x1,y1,x2,y2 手写
        # 这里简单起见，转换回 x,y,w,h 供 OpenCV 使用 (x,y 是左上角)
        opencv_boxes = []
        for box in nms_boxes:
            opencv_boxes.append([int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])])
            
        indices = cv2.dnn.NMSBoxes(opencv_boxes, max_scores.tolist(), self.conf_thres, self.iou_thres)
        
        final_boxes = []
        final_scores = []
        final_classes = []
        
        dw, dh = pad
        
        # 6. 还原坐标到原图尺寸
        if len(indices) > 0:
            # cv2.dnn.NMSBoxes 返回的是 list of list 或者 flat list，兼容处理
            indices = indices.flatten()
            
            for i in indices:
                box = nms_boxes[i] # x1, y1, x2, y2
                
                # 移除 Padding
                box[0] -= dw
                box[1] -= dh
                box[2] -= dw
                box[3] -= dh
                
                # 缩放回原图
                box /= scale
                
                # 边界截断
                box[0] = max(0, box[0])
                box[1] = max(0, box[1])
                box[2] = min(self.img_w, box[2])
                box[3] = min(self.img_h, box[3])
                
                final_boxes.append(box.astype(int))
                final_scores.append(max_scores[i])
                final_classes.append(class_ids[i])
                
        return final_boxes, final_scores, final_classes

    
    def run(self, image_path, v8=False, v10=False):
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return

        # 预处理
        img_data, scale, pad = self.preprocess(img)

        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: img_data})
       
        # 后处理
        det_boxes, det_scores, det_classes = self.postprocess(outputs[0], scale, pad, v8)
        
        # 绘制结果
        print(f"检测到 {len(det_boxes)} 个目标")
        self.draw_results(img, det_boxes, det_scores, det_classes)
    
    def draw_results(self, img, boxes, scores, classes):
        # COCO 类别 (仅作示例，如果是自定义数据集需修改)
        coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            
            # 随机颜色
            color = (0, 255, 0)
            
            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 写标签
            label = f"{coco_names[cls_id] if cls_id < len(coco_names) else cls_id}: {score:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
            cv2.rectangle(img, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), 0, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            
        # 保存或显示
        output_path = "result.jpg"
        cv2.imwrite(output_path, img)
        print(f"结果已保存至: {output_path}")
        # cv2.imshow("Result", img)
        # cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='weights/yolo11n.onnx', help="Path to ONNX model")
    parser.add_argument("--image", type=str, default='data/1.jpg', help="Path to input image")
    args = parser.parse_args()

    runner = YOLO_ONNX_Runner(args.model)
    runner.run(args.image, v8=True)