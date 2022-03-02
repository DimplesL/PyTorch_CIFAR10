# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@contact: qiuyurui@maituan.com
@software: Pycharm
@file: export_onnx.py
@time: 2021/8/25 7:47 下午
@desc:
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from cifar10_models.resnet import resnet18


def file_size(file):
    # Return file size in MB
    return Path(file).stat().st_size / 1e6


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def export_torchscript(model, img, file, optimize):
    # TorchScript model export
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript.pt')
        ts = torch.jit.trace(model, img, strict=False)
        (optimize_for_mobile(ts) if optimize else ts).save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return ts
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_onnx(model, img, file, opset, dynamic, simplify):
    # ONNX model export
    prefix = colorstr('ONNX:')
    try:
        import onnx

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')
        torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(img.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        print(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_coreml(model, img, file):
    # CoreML model export
    prefix = colorstr('CoreML:')
    try:
        import coremltools as ct

        print(f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = file.with_suffix('.mlmodel')
        model.train()  # CoreML exports should be placed in model.train() mode
        ts = torch.jit.trace(model, img, strict=False)  # TorchScript model
        model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        model.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


def attempt_load(model_path, class_num=11, map_location='cpu'):
    ckpt = torch.load(model_path, map_location=map_location)
    model = resnet18(class_num)
    new_state_dict = {k.split('model.')[-1]: v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(new_state_dict)
    return model


def run(weights='./yolov5s.pt',  # weights path
        img_size=(32, 32),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx', 'coreml'),  # include formats
        half=False,  # FP16 half-precision export
        optimize=False,  # TorchScript: optimize for mobile
        dynamic=False,  # ONNX: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        ):
    t = time.time()
    include = [x.lower() for x in include]
    img_size *= 2 if len(img_size) == 1 else 1  # expand
    file = Path(weights)
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # Update model
    if half:
        img, model = img.half(), model.half()  # to FP16
    model.eval()  # training mode = no Detect() layer grid construction

    for _ in range(2):
        y = model(img)  # dry runs
    print(f"\n{colorstr('PyTorch:')} starting from {weights} ({file_size(weights):.1f} MB)")

    # Exports
    if 'torchscript' in include:
        export_torchscript(model, img, file, optimize)
    if 'onnx' in include:
        export_onnx(model, img, file, opset, dynamic, simplify)
    if 'coreml' in include:
        export_coreml(model, img, file)

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s)'
          f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
          f'\nVisualize with https://netron.app')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/Users/qiuyurui/Desktop/models_file/res18focalfixepoch=49-step=1849.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[32, 32], help='image (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='include formats')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('export: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    print(opt)
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
