from ultralytics import YOLO
from ultralytics import YOLOv10
import argparse
import torch
import torch.nn as nn

def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt', help='weights path')
    args_parser.add_argument('--yaml', type=str, default='yolov10s.yaml', help='model yaml file')
    args_parser.add_argument('--sourse', type=str, default='data/1.jpg', help='image/video path')
    args_parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    args_parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    args_parser.add_argument('--img_size', type=int, nargs='+', default=[640,640], help='height and width of the input image')
    args_parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    args_parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    args_parser.add_argument('--save', action='store_true', help='save result image/video')
    args_parser.add_argument('--show_labels', action='store_true', help='show label on result image/video')
    args_parser.add_argument('--show_conf', action='store_true', help='show confidence score on result image/video')
    args_parser.add_argument('--line_width', type=int, default=1, help='bounding box line width')
    args_parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    args_parser.add_argument('--name', default='exp', help='save results to project/name')
    args_parser.add_argument('--v10', action='store_true', help='use yolov10 model for inference')
    args = args_parser.parse_args()
    return args


def run_infer(args):
    if args.v10:
        assert args.yaml, '--yaml must be specified for yolov10 inference'
        model = YOLOv10(args.yaml)
        ckpt = torch.load(args.weights, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        if not isinstance(state_dict, dict):
            state_dict = state_dict.state_dict()
        model.model.load_state_dict(state_dict, strict=True)
        model.model.eval()
    else:
        model = YOLO(args.weights)
        model.fuse()
    if args.half:
        model.to('cuda').half()
    else:
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = model.predict(
        source=args.sourse,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
        imgsz=args.img_size,
        save=args.save,
        project=args.project,
        name=args.name,
        line_width=args.line_width,
        show_labels=args.show_labels,
        show_conf=args.show_conf
    )
    return results


if __name__ == '__main__':
    args = parse_args()
    results = run_infer(args)