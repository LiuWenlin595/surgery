import joblib

import net2net
N2N = net2net.Net2Net()

def print_variables(checkpoint):
    value_dict = joblib.load(checkpoint)
    for name, v in value_dict.items():
        if name == "ppo/strategy/feat_latent_fc/w/Adam:0":
            print("xihaxiha")
            print(v[152])
            print(v[153])
            print(v[154])
            print(v[160])
            print(v[-1])
        # print(name, v.shape)

## 扩展指定layer的输入维度, new_width是新网络层的宽度
def wider_fc_obs(value_dict, layer_name, new_width):
    stem = layer_name.strip(':0')

    fc_name = stem + ':0'
    fc_weight = value_dict[fc_name]
    new_fc_weight = N2N.wider_obs(fc_weight, new_width)
    value_dict[fc_name] = new_fc_weight
    print(f'expand input layer {fc_name}, from {fc_weight.shape} to {new_fc_weight.shape} ')

    for suffix in ['/Adam:0', '/Adam_1:0']:
        adam_name = stem + suffix
        if adam_name not in value_dict:
            print(f'SKIP {adam_name}')
            continue
        adam_weight = value_dict[adam_name]
        new_adam_weight = N2N.padding_for_adam(adam_weight, new_fc_weight.shape)
        value_dict[adam_name] = new_adam_weight
        print(f'expand input layer {adam_name}, from {adam_weight.shape} to {new_adam_weight.shape} ')

    return value_dict

## 复制相同维度的网络结构
def copy_fc_obs(value_dict, layer_name, new_name):
    stem = layer_name.strip(':0')
    fc_name = stem + ':0'
    fc_weight = value_dict[fc_name]
    new_fc_name = new_name + ":0"
    value_dict[new_fc_name] = fc_weight

    return value_dict


## 扩展指定layer的输出维度, idx之前不动, idx之后复制
def wider_fc_out(value_dict, layer_name, idx):
    stem = layer_name.strip(':0')
    fc_name = stem + ':0'
    bias_name = stem.strip('w') + 'b:0'
    fc_weight = value_dict[fc_name]
    fc_bias = value_dict[bias_name]
    new_fc_weight, new_fc_bias = N2N.insert_fc_out(fc_weight, fc_bias, idx)
    value_dict[fc_name] = new_fc_weight
    value_dict[bias_name] = new_fc_bias
    print(f'expand output layer {fc_name}, from {fc_weight.shape} to {new_fc_weight.shape} ')
    print(f'expand output layer {bias_name}, from {fc_bias.shape} to {new_fc_bias.shape} ')

    for suffix in ['/Adam:0', '/Adam_1:0']:
        adam_name = stem + suffix
        adam_bias_name = stem.strip('w') + 'b' + suffix
        if adam_name not in value_dict or adam_bias_name not in value_dict:
            print(f'SKIP {adam_name}')
            continue
        adam_weight = value_dict[adam_name]
        adam_bias = value_dict[adam_bias_name]
        new_adam_weight, new_adam_bias = N2N.insert_fc_out_adam(adam_weight, adam_bias, idx)
        value_dict[adam_name] = new_adam_weight
        value_dict[adam_bias_name] = new_adam_bias
        print(f'expand output layer {adam_name}, from {adam_weight.shape} to {new_adam_weight.shape} ')
        print(f'expand output layer {adam_bias_name}, from {adam_bias.shape} to {new_adam_bias.shape} ')

    return value_dict


def wider_fc_out2(value_dict, layer_name, idx):
    stem = layer_name.strip(':0')
    fc_name = stem + ':0'
    bias_name = stem.strip('weights') + 'biases:0'
    fc_weight = value_dict[fc_name]
    fc_bias = value_dict[bias_name]
    new_fc_weight, new_fc_bias = N2N.insert_fc_out(fc_weight, fc_bias, idx)
    value_dict[fc_name] = new_fc_weight
    value_dict[bias_name] = new_fc_bias
    print(f'expand output layer {fc_name}, from {fc_weight.shape} to {new_fc_weight.shape} ')
    print(f'expand output layer {bias_name}, from {fc_bias.shape} to {new_fc_bias.shape} ')

    for suffix in ['/Adam:0', '/Adam_1:0']:
        adam_name = stem + suffix
        adam_bias_name = stem.strip('weights') + 'biases' + suffix
        if adam_name not in value_dict or adam_bias_name not in value_dict:
            print(f'SKIP {adam_name}')
            continue
        adam_weight = value_dict[adam_name]
        adam_bias = value_dict[adam_bias_name]
        new_adam_weight, new_adam_bias = N2N.insert_fc_out_adam(adam_weight, adam_bias, idx)
        value_dict[adam_name] = new_adam_weight
        value_dict[adam_bias_name] = new_adam_bias
        print(f'expand output layer {adam_name}, from {adam_weight.shape} to {new_adam_weight.shape} ')
        print(f'expand output layer {adam_bias_name}, from {adam_bias.shape} to {new_adam_bias.shape} ')

    return value_dict


if __name__ == '__main__':
    OLD_CHECKPOINT = 'model_402550_old'
    NEW_CHECKPOINT = 'model_402550'
    # print_variables(NEW_CHECKPOINT)
    value_dict = joblib.load(OLD_CHECKPOINT)

    # 扩展输入层的例子
    EXPAND_INPUT_LIST = [('ppo/strategy/feat_latent_fc/w', 163)]

    for layer_name, new_width in EXPAND_INPUT_LIST:
        # print(f'expand layer {layer_name} to {new_width} ')
        value_dict = wider_fc_obs(value_dict, layer_name, new_width)

    ## 复制网络的例子
    # keys = list(value_dict.keys())
    # for name in keys:
    #     # print(name, value_dict[name].shape)
    #     if "vf_latent_low_2" in name:
    #         layer_name = name.strip(':0')
    #         new_layer_name = layer_name.replace("vf_latent_low_2", "vf_latent_low_3")
    #         print(f'copy layer {layer_name} to {new_layer_name} ')
    #         value_dict = copy_fc_obs(value_dict, layer_name, new_layer_name)
    #     elif "diag_vmix_high_3" in name:
    #         layer_name = name.strip(':0')
    #         new_layer_name = layer_name.replace("diag_vmix_high_3", "diag_vmix_high_4")
    #         print(f'copy layer {layer_name} to {new_layer_name} ')
    #         value_dict = copy_fc_obs(value_dict, layer_name, new_layer_name)

    ## 扩展输出层的例子
    EXPAND_OUTPUT_LIST = [('ppo/strategy/predict_network/fully_connected_2/weights', 35)]
    for layer_name, new_width in EXPAND_OUTPUT_LIST:
        # print(f'expand layer {layer_name} to {new_width}')
        value_dict = wider_fc_out2(value_dict, layer_name, new_width)


    print(f'save new checkpoint to {NEW_CHECKPOINT}')
    joblib.dump(value_dict, NEW_CHECKPOINT)

    # print_variables(NEW_CHECKPOINT)
