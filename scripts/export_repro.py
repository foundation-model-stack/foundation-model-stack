from pathlib import Path
import torch
import torch._inductor.config
from torch.export import export, Dim, save, load
import torch._dynamo.config

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(100, 50)
        self.net2 = torch.nn.Linear(50, 100)

    def forward(self, x, y):
        z = self.net1(x)
        z = z[:, -y.shape[0]:, :]
        z = self.net2(z)
        return z

toy_model = ToyModel()

torch._inductor.config.fx_graph_cache = True
print("exporting and compiling model")
export_file = Path("model.pt2")
if not export_file.is_file():
    seqlen = Dim("seqlen", max=4)
    exported_model = export(
        toy_model,
        (torch.randn(4, 5, 100), torch.randn(3)),
        dynamic_shapes={"x": {}, "y": {0: seqlen}}
    )
    save(exported_model, export_file)

# compiling can make first inference pass slow
loaded_model = load(export_file)
print(loaded_model(torch.randn(4, 5, 50), torch.randn(3)))
