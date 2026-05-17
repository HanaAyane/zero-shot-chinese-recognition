import torch
import torch.nn as nn
from model.resnet_v2 import resnet101, resnet18, resnet50
from model.densenet import densenet121, densenet36
from model.resnet45 import resnet45
from model.densenetdsbn import densenet36_dsbn, densenet44_dsbn

class VisualBackbone(nn.Module):
    def __init__(self, args):
        super(VisualBackbone, self).__init__()
        if args.backbone == 'densenet121':
            self.featureExtractor = densenet121(
                first_pooling=args.first_pooling, activation=args.activation)
            decoder_dim = 1024
        elif args.backbone == 'densenet36':
            self.featureExtractor = densenet36(
                first_pooling=args.first_pooling, activation=args.activation)
            decoder_dim = 244
        elif args.backbone == 'densenet36_dsbn':
            self.featureExtractor = densenet36_dsbn()
            decoder_dim = 244
        elif args.backbone == 'densenet44_dsbn':
            self.featureExtractor = densenet44_dsbn()
            decoder_dim = 308
        elif args.backbone == 'resnet50':
            self.featureExtractor = resnet50()
            decoder_dim = 2048
        elif args.backbone == 'resnet101':
            self.featureExtractor = resnet101()
            decoder_dim = 2048
        elif args.backbone == 'resnet45':
            self.featureExtractor = resnet45()
            decoder_dim = 512
        else:
            raise Exception('backbone is not in support list!')

        self.decoder_dim = decoder_dim

    def forward(self, x, char_template):
        # only dsbn
        feat, template_feat = self.featureExtractor([x, char_template])
        return feat, template_feat
