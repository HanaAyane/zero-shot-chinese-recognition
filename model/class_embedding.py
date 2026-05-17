import torch
import torch.nn as nn
from model.transferLayer import TransferLayer

class ClassEmb(nn.Module):
    def __init__(self, args):
        super(ClassEmb, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.001), requires_grad=True)

        # transfer
        self.transferLayer = TransferLayer(args)

    def forward(self, onehot_array, depth_array, part_array, part_embedding, depth_embedding):
        # class embedding
        all_class_embedding = torch.pow(self.alpha, depth_array) * (1 - self.beta * part_array) * onehot_array
        all_class_embedding_org = torch.sum(all_class_embedding, dim=1)
        all_class_embedding_org = torch.cat((all_class_embedding_org, part_embedding, depth_embedding), dim=1)
        # transfer class embedding
        all_class_embedding = self.transferLayer(all_class_embedding_org)

        return all_class_embedding
