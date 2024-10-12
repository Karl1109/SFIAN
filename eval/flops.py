from thop import profile
import torch
from models.SFIAN_network import SFIANNet

if __name__ == '__main__':
    model = SFIANNet(3, 1, 64)
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, (input, ))
    print("flops(G):", flops/1e9, "params(M):",params/1e6)

