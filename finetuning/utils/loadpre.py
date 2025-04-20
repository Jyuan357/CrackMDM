import torch


def load_from(pretrain_dir, net):
    network_name = type(net).__name__
    pretrained_dict = torch.load(pretrain_dir, map_location='cpu')['model']
    model_dict = net.state_dict()
    if any([True if 'encoder.' in k else False for k in pretrained_dict.keys()]):
        pretrained_dict = {k.replace('encoder.', '', 1): v for k, v in pretrained_dict.items() if k.startswith('encoder.')}

    # for k, v in pretrained_dict.items():
    #     print(f"{k}: {v.shape}")
    # print('\n')
    # for k, v in model_dict.items():
    #     print(f"{k}: {v.shape}")

    if network_name  == 'crackformer' or network_name  == 'crackformer_TDAtt':
        droplist = ['head.weight', 'head.bias', 'aux_head.weight', 'aux_head.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.startswith('down')}
        model_dict.update(pretrained_dict)
    
    if network_name  == 'FDNet':
        droplist = ['final_conv.weight', 'final_conv.bias']
        
        prefixes = ['conv5', 'conv4', 'conv3', 'conv2', 'conv1']
        for k in list(pretrained_dict.keys()):
            if any(k.startswith(prefix) for prefix in prefixes):
                droplist.append(k)
        
        mismatched_keys = []
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape != model_dict[k].shape:
                mismatched_keys.append(k)
        droplist.extend(mismatched_keys)
        
    
        for k in droplist:
            del pretrained_dict[k]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('up')}  
        model_dict.update(pretrained_dict)
    
    elif network_name  == 'SwinUNet':
        # droplist = ['output.weight']
        # for k in droplist:
        #     del pretrained_dict[k]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and (k.startswith('patch_embed') or k.startswith('layers.'))}
        model_dict.update(pretrained_dict)

    elif network_name  == 'TransUNet':
        # droplist = ['decoder.conv1.weight', 'decoder.conv1.bias']
        # for k in droplist:
        #     del pretrained_dict[k]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.startswith('encoder.')}
        model_dict.update(pretrained_dict)

    elif network_name  == 'DeepCrackOffical' or network_name  == 'DeepCrack':
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.startswith('down')}
        model_dict.update(pretrained_dict)

    elif network_name  == 'U_Net':
        droplist = ['Conv_1x1.weight', 'Conv_1x1.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('Up')}
        for k in droplist:
            del pretrained_dict[k]
        model_dict.update(pretrained_dict)
    
    elif network_name  == 'UNet_CGA':
        droplist = ['Conv_1x1.weight', 'Conv_1x1.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.startswith('Conv')}
        for k in droplist:
            del pretrained_dict[k]
        model_dict.update(pretrained_dict)

    else:
        raise ValueError(f"Unknown network: {net}")
    
    return model_dict

def load_raw(pretrain_dir, net):
    network_name = type(net).__name__
    pretrained_dict = torch.load(pretrain_dir, map_location='cpu')['model']
    model_dict = net.state_dict()

    for k, v in pretrained_dict.items():
        print(f"{k}: {v.shape}")
    print('\n')
    for k, v in model_dict.items():
        print(f"{k}: {v.shape}")

    return model_dict
    