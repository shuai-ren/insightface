import sys
import onnx
from onnxconverter_common import float16

model_path = sys.argv[1]
model = onnx.load(model_path)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "{}_fp16.onnx".format(model_path[:-5]))
