import  torch
from utils.class_emb_loader import load_class_emb, load_template

def prepare_class_and_template(args, net, numOfGPUs):
    onehot_array, depth_array, part_array, part_embedding, depth_embedding = \
        load_class_emb(args)
    if numOfGPUs > 1:
        all_class_embedding = \
            net.module.class_emb_module(onehot_array.cuda(), depth_array.cuda(), part_array.cuda(), part_embedding.cuda(), depth_embedding.cuda())
    else:
        all_class_embedding = \
            net.class_emb_module(onehot_array.cuda(), depth_array.cuda(), part_array.cuda(), part_embedding.cuda(), depth_embedding.cuda())
    all_class_embedding = all_class_embedding.cpu()
    torch.cuda.empty_cache()

    char_templates = load_template(args)
    image_dummy = torch.FloatTensor(1, char_templates.shape[1], char_templates.shape[2], char_templates.shape[3])
    image_dummy = image_dummy.cuda()

    mini_batch = 512
    mini_bs = char_templates.shape[0] // mini_batch
    if mini_bs == 0:
        char_templates_list = [char_templates]
    else:
        char_templates_list = [char_templates[idx * mini_batch : (idx+1) * mini_batch] for idx in range(mini_bs)]
        char_templates_list.append(char_templates[mini_bs*mini_batch:])

    template_feat_list = [] 
    for char_template in char_templates_list:
        if numOfGPUs > 1:
            _, template_feat = net.module.visual_module(image_dummy, char_template.cuda())
        else:
            _, template_feat = net.visual_module(image_dummy, char_template.cuda())
        template_feat = template_feat.cpu()
        torch.cuda.empty_cache()
        template_feat_list.append(template_feat)

    template_feat = torch.cat(template_feat_list, dim=0)

    if numOfGPUs > 1:
        all_class_embedding = all_class_embedding.expand(numOfGPUs, *all_class_embedding.shape)
        template_feat = template_feat.expand(numOfGPUs, *template_feat.shape)

    return all_class_embedding, template_feat