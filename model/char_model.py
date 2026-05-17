import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import ZSDecoder
from model.class_embedding import ClassEmb
from model.visual_backbone import VisualBackbone
import numpy as np
import random
from utils.class_emb_loader import load_class_emb, load_template


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seen_num = args.seen_classes

        # visual backbone
        self.visual_module = VisualBackbone(args)
        decoder_dim = self.visual_module.decoder_dim

        # class embeddings
        self.onehot_array, self.depth_array, self.part_array, self.part_embedding, self.depth_embedding = \
            load_class_emb(args)

        self.class_emb_module = ClassEmb(args)

        # template
        self.char_template = load_template(args)
        self.conv1x1 = nn.Sequential(nn.Conv2d(decoder_dim, 1, (1, 1)), nn.Sigmoid())
        self.fc = nn.Linear(decoder_dim * 2, decoder_dim)

        self.cosine_sim1 = CosineDistance()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        # decoder
        self.zsDecoder = ZSDecoder(decoder_embedding=decoder_dim,
                                   n_head=4,
                                   dim_feedforward=2048,
                                   num_layers_decoder=1)

    def seen_sampling(self, onehot_array, depth_array, part_array,
                      part_embedding, depth_embedding, char_template,
                      seen_num):
        onehot_array = onehot_array[:seen_num]
        depth_array = depth_array[:seen_num]
        part_array = part_array[:seen_num]
        part_embedding = part_embedding[:seen_num]
        depth_embedding = depth_embedding[:seen_num]
        char_template = char_template[:seen_num]
        return onehot_array, depth_array, part_array, part_embedding, \
            depth_embedding, char_template

    def label_sampler(self,
                      onehot_array,
                      depth_array,
                      part_array,
                      part_embedding,
                      depth_embedding,
                      char_template,
                      char_id,
                      n_max=512):

        if onehot_array.shape[0] > n_max:
            c_label = []
            c_train = np.arange(0, onehot_array.shape[0]).tolist()
            device = char_id.device
            char_id = char_id.view(-1).cpu().data.numpy().tolist()

            for idx in char_id:
                if idx not in c_label:
                    c_label.append(idx)
                    c_train.pop(c_train.index(idx))

            c_neg_num = n_max - len(c_label)
            c_neg = random.choices(c_train, k=c_neg_num)

            c_batch = c_label + c_neg
            c_batch = sorted(c_batch)

            new_char_id = []
            for idx in char_id:
                new_char_id.append(c_batch.index(idx))
            new_char_id = torch.LongTensor(new_char_id).unsqueeze(1).to(device)

            onehot_array = onehot_array[c_batch]
            depth_array = depth_array[c_batch]
            part_array = part_array[c_batch]
            part_embedding = part_embedding[c_batch]
            depth_embedding = depth_embedding[c_batch]
            char_template = char_template[c_batch]
            char_id = new_char_id

        return onehot_array, depth_array, part_array, part_embedding, \
            depth_embedding, char_template, char_id

    def forward(self, x, char_id, sampling=False, training=True):
        if training:
            return self.forward_train(x, char_id, sampling)
        else:
            image, template_feat, all_class_embedding = x[0], x[1], x[2]
            return self.forward_test(image, template_feat, all_class_embedding)

    def forward_train(self, x, char_id, sampling=False):
        # load
        onehot_array = self.onehot_array
        depth_array = self.depth_array
        part_array = self.part_array
        part_embedding = self.part_embedding
        depth_embedding = self.depth_embedding
        char_template = self.char_template

        if sampling:
            onehot_array, depth_array, part_array, part_embedding, depth_embedding, char_template = \
                self.seen_sampling(onehot_array, depth_array, part_array, part_embedding,
                depth_embedding, char_template, self.seen_num)

            onehot_array, depth_array, part_array, part_embedding, depth_embedding, char_template, char_id = \
                self.label_sampler(onehot_array, depth_array, part_array, part_embedding,
                depth_embedding, char_template, char_id, n_max=512)

        onehot_array, depth_array, part_array = onehot_array.to(
            x.device), depth_array.to(x.device), part_array.to(x.device)
        part_embedding, depth_embedding = part_embedding.to(
            x.device), depth_embedding.to(x.device)
        char_template = char_template.to(x.device)

        # class embedding
        all_class_embedding = self.class_emb_module(onehot_array, depth_array, part_array, part_embedding, depth_embedding)

        # visual feat
        feat, template_feat = self.visual_module(x, char_template)
        spatial_attn = self.conv1x1(template_feat)
        template_feat = template_feat.flatten(2)
        spatial_attn = spatial_attn.flatten(2).permute(0, 2, 1)
        template_feat = torch.bmm(template_feat, spatial_attn)
        template_feat = template_feat.squeeze(2)

        # DCE loss
        feat_pool = self.AdaptiveAvgPool(feat)
        feat_pool = feat_pool.view(feat_pool.shape[0], -1)
        template_matrix = self.cosine_sim1(feat_pool, template_feat)

        # pl loss
        pl_loss = F.mse_loss(feat_pool, template_feat[char_id.view(-1)])

        # transformer
        fuse_embed = torch.cat((all_class_embedding, template_feat), dim=1)
        fuse_embed = self.fc(fuse_embed)

        pred, attn_map = self.zsDecoder(feat, fuse_embed)
        return {
            'pred': pred,
            'attn_map': attn_map,
            'char_id': char_id,
            'template_matrix': template_matrix,
            'pl_loss': pl_loss
        }

    def forward_test(self, image, template_feat, all_class_embedding):
        if len(template_feat.shape) > 4 and len(all_class_embedding.shape) > 2:
            template_feat = template_feat.squeeze(0)
            all_class_embedding = all_class_embedding.squeeze(0)

        template_dummy = torch.FloatTensor(1, image.shape[1], image.shape[2],
                                           image.shape[3])
        template_dummy = template_dummy.to(image.device)
        feat, _ = self.visual_module(image, template_dummy)
        spatial_attn = self.conv1x1(template_feat)
        template_feat = template_feat.flatten(2)
        spatial_attn = spatial_attn.flatten(2).permute(0, 2, 1)
        template_feat = torch.bmm(template_feat, spatial_attn)
        template_feat = template_feat.squeeze(2)
        # transformer
        fuse_embed = torch.cat((all_class_embedding, template_feat), dim=1)
        fuse_embed = self.fc(fuse_embed)
        pred, attn_map = self.zsDecoder(feat, fuse_embed)
        return {'pred': pred, 'feat': feat}


class CosineDistance(nn.Module):

    def __init__(self):
        super(CosineDistance, self).__init__()
        self.scale_weight = nn.Parameter(torch.tensor(10.0),
                                         requires_grad=True)

    def distance(self, features, y_embedding):
        features_norm = torch.norm(features, p=2, dim=1, keepdim=True)
        y_embedding = torch.transpose(y_embedding, 0, 1)
        y_embedding_norm = torch.norm(y_embedding, p=2, dim=0, keepdim=True)
        return features @ y_embedding / (features_norm @ y_embedding_norm)

    def forward(self, features, all_class_embedding):
        dist = self.distance(features, all_class_embedding)
        logits = self.scale_weight * dist
        return logits


class L2Distance(nn.Module):

    def __init__(self):
        super(L2Distance, self).__init__()
        self.scale_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def distance(self, features, y_embedding):
        dist = torch.norm(features.unsqueeze(1) - y_embedding.unsqueeze(0),
                          p=2,
                          dim=2)
        return dist

    def forward(self, features, all_class_embedding):
        dist = self.distance(features, all_class_embedding)
        logits = self.scale_weight * dist
        return logits


class WholeModel(nn.Module):

    def __init__(self, args):
        super(WholeModel, self).__init__()
        self.args = args
        self.featureExtractor = densenet121(first_pooling=args.first_pooling,
                                            activation=args.activation)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Linear(1024, args.class_nums)

    def forward(self, x):
        feat = self.featureExtractor(x)
        feat = self.AdaptiveAvgPool(feat)
        feat = feat.view(x.shape[0], -1)
        feat = self.cls(feat)
        return feat


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()