import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

srcs = "/home/aa940249tw/nctu/ml_chip/Lab_2/lab2-1/srcs/"

def load_param(conv_num, fc_num):
    parameters = {}
    for i in range(1, conv_num + 1):
        parameters['W' + str(i)] = np.load(srcs + "Conv" + str(i) + "_weights.npy")
        parameters['b' + str(i)] = np.load(srcs + "Conv" + str(i) + "_bias.npy")
    
    for i in range(conv_num + 1, conv_num + 1 + fc_num):
        parameters['W' + str(i)] = np.load(srcs + "Fc" + str(i) + "_weights.npy")
        parameters['b' + str(i)] = np.load(srcs + "Fc" + str(i) + "_bias.npy")
    return parameters

class vgg16:
    def __init__(self, params):
        self.params = params

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A
    
    def zero_padding(self, X, pad):
        X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
        return X_pad
    
    def linear_foward(self, A_prev, W, b):
        b = np.reshape(b, (b.shape[0], 1))
        Z = np.dot(W, A_prev) + b
        print(Z.shape)
        assert (Z.shape == (W.shape[0], A_prev.shape[1]))
        return Z
    
    def max_pooling(self, A_prev, hparameters = {'f':2, 'stride':2}):
        (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
        f = hparameters["f"]
        stride = hparameters["stride"]
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        A = np.zeros((m, n_C, n_H, n_W)) 
        for i in range(m):
            a_prev = A_prev[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        v_start = h * stride
                        v_end = v_start + f
                        h_start =  w * stride
                        h_end = h_start + f

                        a_prev_slice = a_prev[c, v_start:v_end, h_start:h_end]
                        A[i, c, h, w] = np.max(a_prev_slice)
        return A

    def conv_single(self, a_slice_prev, w, b):
        s = np.multiply(a_slice_prev, w)
        Z = np.sum(s) + b
        return self.relu(Z)
    
    def conv_foward(self, A_prev, W, b, hparameters = {'stride':1, 'pad':1}):
        (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
        (n_C, n_C_prev, f, f) = W.shape
        stride = hparameters['stride']
        pad = hparameters['pad']
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
        Z = np.zeros((m, n_C, n_H, n_W))
        A_prev_pad = self.zero_padding(A_prev, pad)

        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        v_start = h * stride
                        v_end = v_start + f
                        h_start =  w * stride
                        h_end = h_start + f
                        a_slice_prev = a_prev_pad[:, v_start:v_end, h_start:h_end]
                        Z[i, c, h, w] = self.conv_single(a_slice_prev, W[c, ...], b[c, ...])
    
        assert(Z.shape == (m, n_C, n_H, n_W))
        return Z

    def foward(self, x):
        print('Conv1...')
        out = self.conv_foward(x, self.params['W1'], self.params['b1'])
        print('Conv2...')
        out = self.conv_foward(out, self.params['W2'], self.params['b2'])
        out = self.max_pooling(out)
        print('Conv3...')
        out = self.conv_foward(out, self.params['W3'], self.params['b3'])
        print('Conv4...')
        out = self.conv_foward(out, self.params['W4'], self.params['b4'])
        out = self.max_pooling(out)
        print('Conv5...')
        out = self.conv_foward(out, self.params['W5'], self.params['b5'])
        print('Conv6...')
        out = self.conv_foward(out, self.params['W6'], self.params['b6'])
        print('Conv7...')
        out = self.conv_foward(out, self.params['W7'], self.params['b7'])
        out = self.max_pooling(out)
        print('Conv8...')
        out = self.conv_foward(out, self.params['W8'], self.params['b8'])
        print('Conv9...')
        out = self.conv_foward(out, self.params['W9'], self.params['b9'])
        print('Conv10...')
        out = self.conv_foward(out, self.params['W10'], self.params['b10'])
        out = self.max_pooling(out)
        print('Conv11...')
        out = self.conv_foward(out, self.params['W11'], self.params['b11'])
        print('Conv12...')
        out = self.conv_foward(out, self.params['W12'], self.params['b12'])
        print('Conv13...')
        out = self.conv_foward(out, self.params['W13'], self.params['b13'])
        out = self.max_pooling(out)
        out = torch.from_numpy(out)
        out = F.adaptive_avg_pool2d(out, [7, 7]).detach().cpu()
        out = out.reshape(-1, out.size(0)).numpy()
        print(out.shape)
        print('FC1...')
        out = self.linear_foward(out, self.params['W14'], self.params['b14'])
        out = self.relu(out)
        print('FC2...')
        out = self.linear_foward(out, self.params['W15'], self.params['b15'])
        out = self.relu(out)
        print('FC3...')
        out = self.linear_foward(out, self.params['W16'], self.params['b16'])
        return out

def VGG16():
    params = load_param(13, 3)
    params['W1'].shape
    return vgg16(params)

def load_input():
    img_to_tensor = transforms.ToTensor()
    img = Image.open('srcs/input.jpg')
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)
    tensor = tensor.resize_(1,3,224,224)
    return tensor.detach().cpu().numpy()

if __name__ == "__main__":
    model = VGG16()
    input = load_input()
    ans = model.foward(input)
    ans = np.squeeze(ans, axis = 1)
    print(ans)

