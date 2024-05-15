"""An example script showing how to export a flow to ONNX format."""

import pyrootutils

root = pyrootutils.setup_root(search_from=".", pythonpath=True)

import torch as T
import torch.nn as nn
from mltools.mltools.flows import rqs_flow

flow = rqs_flow(
    xz_dim=2,
    ctxt_dim=1,
    num_stacks=1,
    mlp_width=32,
    mlp_depth=1,
    mlp_act=nn.ReLU,
    tail_bound=4,
    dropout=0.1,
    num_bins=4,
    do_lu=False,
    do_norm=False,
    flow_type="coupling",
    init_identity=True,
)


# Export the sampling model using ONNX runtime
class FlowWrapper(nn.Module):
    """Basic wrapper for the sampler method of a Flow."""

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, c: T.Tensor) -> T.Tensor:
        return flow.sample(c.shape[0], c)[0]


# Move to CPU and eval
flow.eval()
model = FlowWrapper(flow)
c = T.randn(1, 1)

T.onnx.export(
    model=model,
    args=c,
    f="flow.onnx",
    export_params=True,
    verbose=True,
    input_names=["context"],
    dynamic_axes={"context": {0: "batch_size"}},
    output_names=["samples"],
    opset_version=16,
)

import onnxruntime

ort_session = onnxruntime.InferenceSession("flow.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: c.numpy()}
ort_outs = ort_session.run(["samples"], ort_inputs)
