# coding=utf-8
from utils.load_checkpoint import Wrapper_yolo
import torch
import onnx
import onnx_graphsurgeon as gs
from io import BytesIO
from utils.events import LOGGER
from utils.end2end import End2End
import argparse
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov10s.pt', help='weights path')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--topk_all', type=int, default=100, help='max number of detections per image')
    parser.add_argument('--iou_thres', type=float, default=0.7, help='iou threshold for NMS')
    parser.add_argument('--conf_thres', type=float, default=0.45, help='confidence threshold for NMS')
    parser.add_argument('--dynamic_batch', action='store_true', help='whether to export dynamic batch size')
    parser.add_argument('--end2end', action='store_true', help='whether to export end2end model')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640,640], help='height and width of the input image')
    parser.add_argument('--device', default='cpu', help='device to use for export')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    parser.add_argument('--ultralytics', action='store_true', help='whether the model is from ultralytics')
    parser.add_argument('--simplify', action='store_false', help='whether to simplify onnx model using onnxsim')
    opt = parser.parse_args()
    return opt

def run_export(opt):
    device = torch.device(opt.device)
    LOGGER.info("Loading model...")
    # model = load_checkpoint(opt.weights,ultralytics=opt.ultralytics, map_location=device)
    if opt.end2end and "yolov10" in opt.weights:
        raise NotImplementedError("End2End export for YOLOv10 is not supported.")
    if opt.ultralytics:
        model = YOLO(opt.weights).model
        model = Wrapper_yolo(model)
    elif "yolov9" in  opt.weights:
        pass
    elif "yolov7" in opt.weights:
        pass
    elif "yolov6" in opt.weights:
        pass
    else:
        raise NotImplementedError("Input model are supported now.")
    model = model.to(device)
    model.eval()
    if len(opt.imgsz) == 1:
        opt.imgsz = [opt.imgsz[0], opt.imgsz[0]]
    img = torch.randn(opt.batch, 3, opt.imgsz[0], opt.imgsz[1]).to(device)
    dynamic_axes = None
    if opt.dynamic_batch:
        dynamic_axes = {
            'images': {
                0: 'batch',
            }, }
        if opt.end2end:
            output_axes = {
                'num_dets': {0: 'batch'},
                'det_boxes': {0: 'batch'},
                'det_scores': {0: 'batch'},
                'det_classes': {0: 'batch'},
            }
        else:
            output_axes = {
                'outputs': {0: 'batch'},
            }
        dynamic_axes.update(output_axes)
    if opt.end2end and not "yolov9" in opt.weights:
        LOGGER.info("Adding End2End (NMS) layers...")
        model = End2End(model, ultralytics=opt.ultralytics, max_obj=opt.topk_all, iou_thres=opt.iou_thres, score_thres=opt.conf_thres,
                        device=device, ort=False, with_preprocess=False)
    elif opt.end2end and "yolov9" in  opt.weights:
        LOGGER.info("Adding End2End (NMS) layers for YOLOv9...")
        model = End2End(model, ultralytics=opt.ultralytics, max_obj=opt.topk_all, iou_thres=opt.iou_thres, score_thres=opt.conf_thres,
                        device=device, ort=False, with_preprocess=False, v9=True)
    try:
        LOGGER.info('\nStarting to export ONNX...')
        export_file = opt.weights.replace('.pt', '.onnx')  # filename
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes'] if opt.end2end else ['outputs']
        with BytesIO() as f:
            torch.onnx.export(model, img, f, verbose=False, opset_version=opt.opset,
                            training=torch.onnx.TrainingMode.EVAL,
                            do_constant_folding=True,
                            input_names=['images'],
                            dynamo=False,
                            output_names=output_names, 
                            dynamic_axes=dynamic_axes)
            f.seek(0)
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            LOGGER.info("Optimizing graph with onnx-graphsurgeon...")
            graph = gs.import_onnx(onnx_model)
            graph.cleanup().toposort()  #从图形中删除未使用的节点和张量，并对图形进行拓扑排序
            # Shape Estimation
            estimated_graph = None
            try:
                # 即使是大模型，使用 export_onnx 生成 proto 也可能比较安全，但 infer_shapes 偶尔会失败
                estimated_graph = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
            except Exception as e:
                LOGGER.warning(f"Shape inference failed, saving without updated shapes: {e}")
                estimated_graph = gs.export_onnx(graph)
            
            if opt.simplify:
                LOGGER.info("Simplifying with onnx-simplifier...")
                try:
                    import onnxsim
                    model_simp, check = onnxsim.simplify(estimated_graph)
                    if check:
                        estimated_graph = model_simp
                        LOGGER.info("Simplification successful.")
                    else:
                        LOGGER.warning("Simplification check failed. Saving unsimplified model.")
                except Exception as e:
                    LOGGER.warning(f"Simplification process error: {e}")
            onnx.save(estimated_graph, export_file)
            LOGGER.info(f'ONNX export success: {export_file}')
    except Exception as e:
        LOGGER.info(f'ONNX export failure: {e}')
        raise e



if __name__ == "__main__":
    opt = parse_opt()
    run_export(opt)
