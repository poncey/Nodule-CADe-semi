# coding=utf-8

import torch
import numpy as np


class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len # =144
        self.max_stride = max_stride # =16
        self.stride = stride # =4
        self.margin = margin # =32
        self.pad_value = pad_value # =170
        
    def split(self, data, side_len = None, max_stride = None, margin = None): # 将肺部扫描（抠出肺部之后的）分割为208*208*208的图块，每个图块重叠区域为32像素
        if side_len is None:
            side_len = self.side_len
        if max_stride is None:
            max_stride = self.max_stride
        if margin is None:
            margin = self.margin
        
        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        
        nzhw = [nz,nh,nw]
        self.nzhw = nzhw
        
        pad = [ [0, 0],
                [margin, nz * side_len - z + margin],
                [margin, nh * side_len - h + margin],
                [margin, nw * side_len - w + margin]]
        data = np.pad(data, pad, 'edge') # 用边缘值填充

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len # 208*208*208的图块之间重叠区域为32像素，相当于208*208*208的裁剪框以144的步长在padding过后的肺部图像上移动
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0) # splits的形状为[分割的图块数=nz*nh*nw，208,208,208]，即当前肺部扫描所分割出的所有图块
        return splits,nzhw # 

    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw is None:
            print '-'*100,'nzhw isNone'
            #nz = self.nz
            #nh = self.nh
            #nw = self.nw
        else:
            nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3], # anchor数
            splits[0].shape[4]), np.float32) # 5

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output 
