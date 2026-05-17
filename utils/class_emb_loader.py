import numpy as np
import torch
import pickle as pkl
 
def load_class_emb(args):   
    data = np.load(args.emb_path)

    onehot_array, depth_array, part_array = data['onehot_array'], data[
        'depth_array'], data['part_array']
    onehot_array = torch.FloatTensor(onehot_array)
    depth_array = torch.FloatTensor(depth_array)
    part_array = torch.FloatTensor(part_array)

    part_array_norm = norm(part_array + 1)
    part_embedding = torch.sum((part_array_norm * onehot_array),
                                    dim=1,
                                    dtype=torch.float32)
    depth_array_norm = norm(depth_array + 1)
    depth_embedding = torch.sum((depth_array_norm * onehot_array),
                                        dim=1,
                                        dtype=torch.float32)
    
    return onehot_array, depth_array, part_array, part_embedding, depth_embedding

def norm(x):
    return x / torch.max(x)

def load_template(args):
    f = open(args.template_path, 'rb')
    char_template = pkl.load(f)['char']
    char_template = torch.FloatTensor([item.data.numpy() for item in char_template])
    return char_template