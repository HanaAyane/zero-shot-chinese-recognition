import torch
import torch.nn as nn

class TransferLayer(nn.Module):

    def __init__(self, args):
        super(TransferLayer, self).__init__()
        emb_dim = args.emb_dim
        if args.backbone == 'resnet50' or args.backbone == 'resnet101':
            self.fc = nn.Linear(emb_dim * 3, 2048)
        elif args.backbone == 'densenet121':
            self.fc = nn.Linear(emb_dim * 3, 1024)
        elif args.backbone == 'resnet45':
            self.fc = nn.Linear(emb_dim * 3, 512)
        elif args.backbone == 'densenet36' or args.backbone == 'densenet36_dsbn':
            self.fc = nn.Linear(emb_dim * 3, 244)
        elif args.backbone == 'densenet44_dsbn':
            self.fc = nn.Linear(emb_dim * 3, 308)

    def forward(self, class_emb):
        class_emb = self.fc(class_emb)
        return class_emb
