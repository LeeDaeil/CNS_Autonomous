from Pytorch_A3C.Main_Fast_Keras_Porba import MainNet
import numpy as np

temp = MainNet(net_type='LSTM', input_pa=6, output_pa=9, time_leg=10)

# print(temp.fully.count_params())
# temp_w = temp.fully.get_weights()
# for layer_w in temp_w:
#     print(np.shape(layer_w), type(layer_w))
# print('=' * 10)
# mu, sigma = 0, 0.1
# new_w = []
# for layer_w in temp_w:
#     layer_new_w = layer_w + np.random.normal(mu, sigma, size=np.shape(layer_w))
#     print(np.shape(layer_w), np.shape(layer_new_w))
#     new_w.append(layer_new_w)
#
# temp.fully.set_weights(new_w)

print(temp.fully.trainable_weights)
temp.fully.trainable = False
print(temp.fully.trainable_weights)
temp.fully.trainable = True
print(temp.fully.trainable_weights)
