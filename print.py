import joblib

def print_variables(checkpoint):
    value_dict = joblib.load(checkpoint)
    # for name, v in value_dict.items():
    #     print(name, v.shape)
    # for k in sorted(value_dict):
        # print(k, value_dict[k].shape)
    # print("len:", len(value_dict.keys()))
    print(1)
    print(value_dict["ppo/strategy/feat_latent_fc/w/Adam:0"])
    print(2)
    print(value_dict["ppo/strategy/feat_latent_fc/w/Adam_1:0"])
    print(3)
    print(value_dict["ppo/strategy/feat_latent_fc/w:0"])
    print(4)
    print(value_dict["ppo/strategy/predict_network/fully_connected_2/biases/Adam:0"])
    print(5)
    print(value_dict["ppo/strategy/predict_network/fully_connected_2/biases/Adam_1:0"])
    print(6)
    print(value_dict["ppo/strategy/predict_network/fully_connected_2/biases:0"])
    print(7)
    print(value_dict["ppo/strategy/predict_network/fully_connected_2/weights/Adam:0"])
    print(8)
    print(value_dict["ppo/strategy/predict_network/fully_connected_2/weights/Adam_1:0"])
    print(9)
    print(value_dict["ppo/strategy/predict_network/fully_connected_2/weights:0"])

# tf.layers.dense：kernel/bias
# fc()：w/b
# tf.contrib.layers.fully_connected：weights/biases

if __name__ == '__main__':
    CHECKPOINT = 'model_402550'
    print_variables(CHECKPOINT)