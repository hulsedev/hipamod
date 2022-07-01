import os
from typing import Union, Optional
from pathlib import Path

from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    AutoFeatureExtractor,
)
from transformers.onnx.features import FeaturesManager
from transformers.onnx.utils import get_preprocessor


from latency.pilot.opt.compress.export_onnx import OPTOnnxConfig


class OPTORTQuantizer(ORTQuantizer):
    def __init__(
        self,
        preprocessor: Union[AutoTokenizer, AutoFeatureExtractor],
        model: PreTrainedModel,
        feature: str = "default",
        opset: Optional[int] = None,
    ):
        # super().__init__(
        #    GPT2Tokenizer.from_pretrained("gpt2"),
        #    GPT2Model.from_pretrained("gpt2"),
        #    feature="causal-lm",
        # )

        self.preprocessor = preprocessor
        self.model = model

        self.feature = feature
        self._model_type = model.config.model_type
        self._onnx_config = OPTOnnxConfig(self.model.config)
        self.opset = self._onnx_config.default_onnx_opset if opset is None else opset

        self._calibrator = None

    @staticmethod
    def from_pretrained(
        model_name_or_path: Union[str, os.PathLike],
        feature: str,
        opset: Optional[int] = None,
    ) -> "OPTORTQuantizer":
        preprocessor = get_preprocessor(model_name_or_path)
        model_class = FeaturesManager.get_model_class_for_feature(feature)
        model = model_class.from_pretrained(model_name_or_path)

        return OPTORTQuantizer(preprocessor, model, feature, opset)


def main(model_name):
    model_dir = Path(f"latency/pilot/opt/model/{model_name}/")
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    model_ckpt = f"facebook/{model_name}"
    onnx_path = Path(f"latency/pilot/opt/model/{model_name}/model.onnx")
    output_file = Path("model_quantized.onnx")

    quantizer = OPTORTQuantizer.from_pretrained(model_ckpt, feature="causal-lm")
    qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
    quantizer.export(
        onnx_model_path=onnx_path,
        onnx_quantized_model_output_path=model_dir.joinpath(output_file),
        quantization_config=qconfig,
    )
    quantizer.model.config.save_pretrained(model_dir)
    print("Done quantizing model.")


if __name__ == "__main__":
    main("opt-350m")
