from torch import nn
import torch

import torch.nn.functional as F

import numpy as np


class KeyHeatModule_test2(nn.Module):
    def __init__(self, num_classes, in_channel, height, width, group_num):
        super(KeyHeatModule_test2, self).__init__()
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.group_num = group_num
        if group_num['head'] != 0:
            self.head_emb = nn.Linear(in_features=num_classes * width * height * 10, out_features=group_num['head']*2*width)
        if group_num['arm'] != 0:
            self.arm_emb = nn.Linear(in_features=num_classes * width * height * 10, out_features=group_num['arm']*2*width)
        if group_num['upper'] != 0:
            self.upper_emb = nn.Linear(in_features=num_classes * width * height * 10, out_features=group_num['upper']*2*width)
        if group_num['lower'] != 0:
            self.lower_emb = nn.Linear(in_features=num_classes * width * height * 10, out_features=group_num['lower']*2*width)
        if group_num['foot'] != 0:
            self.foot_emb = nn.Linear(in_features=num_classes * width * height * 10, out_features=group_num['foot']*2*width)
        
        self.softmax = nn.Softmax(dim=2)
        
        self.head_key_index = [0, 1, 3, 4]
        self.arm_key_index = [7, 8, 9, 10]
        self.upper_key_index = [5, 6, 11, 12]
        self.lower_key_index = [11, 12, 13, 14]
        self.foot_key_index = [15, 16]

        self.scale = 0.3

    def gen_map_test(self, x, keypoint):
        bs = x.shape[0]
        map = []
        x = self.softmax(x)
        x = torch.exp(x)
        
        for i in range(bs):
            zeros = torch.zeros(size=[x.shape[1], self.width]).cuda()
            
            x1 = x[i, :, 0].clone()
            x1[:, :max(torch.min(keypoint[i, :, 0])-1, 0)] = 0
            x1 = x1 / (x1.sum(-1).unsqueeze(-1))
            x1 = torch.cumsum(x1, dim=-1)
            x1 = torch.where(x1>=self.scale, x1, zeros)

            x2 = x[i, :, 0].clone()
            x2[:, min(torch.max(keypoint[i, :, 0])+1, 6):] = 0
            x2 = x2 / (x2.sum(-1).unsqueeze(-1))
            x2 = torch.cumsum(x2, dim=-1)
            x2 = 1-x2
            x2 = torch.where(x2>=self.scale, x2, zeros)
            
            x_mask = x1 * x2
            x_mask = x_mask.reshape((-1, 1, self.width))
            y1 = x[i, :, 1].clone()
            y1[:, :max(torch.min(keypoint[i, :, 1])-1, 0)] = 0
            y1 = y1 / (y1.sum(-1).unsqueeze(-1))
            y1 = torch.cumsum(y1, dim=-1)
            y1 = torch.where(y1>=self.scale, y1, zeros)
            


            y2 = x[i, :, 1].clone()
            y2[:, min(torch.max(keypoint[i, :, 1])+1, 6):] = 0
            y2 = y2 / (y2.sum(-1).unsqueeze(-1))
            y2 = torch.cumsum(y2, dim=-1)
            y2 = 1-y2
            y2 = torch.where(y2>=self.scale, y2, zeros)

            y_mask = y1 * y2
            y_mask = y_mask.reshape((-1, 1, self.width))
            y_mask = y_mask.transpose(2,1)

            map_t = torch.multiply(y_mask, x_mask)
            map_t = map_t/(torch.max(map_t).item() +1e-7)
            map.append(map_t)
        map = torch.stack(map, axis=0)
        
        return map
        

    def forward(self, x, keypoint):
        bs = x.shape[0]
        keypoint = (keypoint * 7).floor().int()
        keypoint[keypoint>6] = 6

        head_point = keypoint[:, self.head_key_index]
        arm_point = keypoint[:, self.arm_key_index]
        upper_point = keypoint[:, self.upper_key_index]
        lower_point = keypoint[:, self.lower_key_index]
        foot_point = keypoint[:, self.foot_key_index]

        embedding = torch.reshape(x, shape=(-1, self.num_classes*10*self.width*self.height))


        if self.group_num['head'] != 0:
            head_vec = self.head_emb(embedding)
            head_vec = torch.reshape(head_vec, shape=(bs, -1, 2, self.width))
            head_map = self.gen_map_test(head_vec, head_point).view((bs, -1, 1, self.width, self.height))
        else:
            head_map = None

        if self.group_num['arm'] != 0:
            arm_vec = self.arm_emb(embedding)
            arm_vec = torch.reshape(arm_vec, shape=(bs, -1, 2, self.width))
            arm_map = self.gen_map_test(arm_vec, arm_point).view((bs, -1, 1, self.width, self.height))
        else:
            arm_map = None

        if self.group_num['upper'] != 0:
            upper_vec = self.upper_emb(embedding)
            upper_vec = torch.reshape(upper_vec, shape=(bs, -1, 2, self.width))
            upper_map = self.gen_map_test(upper_vec, upper_point).view((bs, -1, 1, self.width, self.height))
        else:
            upper_map = None

        if self.group_num['foot'] != 0:
            foot_vec = self.foot_emb(embedding)
            foot_vec = torch.reshape(foot_vec, shape=(bs, -1, 2, self.width))
            foot_map = self.gen_map_test(foot_vec, foot_point).view((bs, -1, 1, self.width, self.height))
        else:
            foot_map = None

        if self.group_num['lower'] != 0:
            lower_vec = self.lower_emb(embedding)
            lower_vec = torch.reshape(lower_vec, shape=(bs, -1, 2, self.width))
            lower_map = self.gen_map_test(lower_vec, lower_point).view((bs, -1, 1, self.width, self.height))
        else:
            lower_map = None
        
        return head_map, arm_map, upper_map, lower_map, foot_map




import torch.nn.modules.conv as conv


class AddCoords_test(nn.Module):
    def __init__(self, num_classes):
        super(AddCoords_test, self).__init__()
        channel_num = 17*3+2
        
        self.linear = nn.Sequential(nn.Conv2d(in_channels=17*3+2, out_channels=num_classes, kernel_size=1,
                                   stride=1, bias=False),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1,
                                   stride=1, bias=False),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1,
                                   stride=1, bias=False),
                                   nn.ReLU(inplace=False),
                                    nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1,
                                   stride=1, bias=False),
        )

    def forward(self, input_tensor, keypoints):

        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        input_tensor = input_tensor.cuda()
        xx_channel = xx_channel.cuda()
        yy_channel = yy_channel.cuda()
        out = torch.cat([xx_channel, yy_channel], dim=1)

        for i in range(keypoints.shape[1]):
            
            xk = (keypoints[:, i, 0]*dim_x/(dim_x -1))*2-1
            xk = xk.reshape(batch_size_shape, 1, 1, 1)
            xk = xk.repeat(1, 1, dim_y, dim_x)
            
            yk = (keypoints[:, i, 1]*dim_y/(dim_y -1))*2-1
            yk = yk.reshape(batch_size_shape, 1, 1, 1)
            yk = yk.repeat(1, 1, dim_y, dim_x)
            rr = torch.sqrt(torch.pow(xx_channel - xk, 2) + torch.pow(yy_channel - yk, 2))
            out = torch.cat([out, xk, yk, rr], dim=1)
        out = self.linear(out)
        out = torch.cat([input_tensor, out], dim=1)

        return out



class CoordConv2d_test(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(CoordConv2d_test, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords_test(num_classes)
        self.conv = nn.Conv2d(in_channels + num_classes, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor, keypoints):
        out = self.addcoords(input_tensor, keypoints)
        out = self.conv(out)

        return out



class Framework_merge_base(nn.Module):

    def __init__(self, backbone, num_classes, in_channel, height, width, in_channel2, height2, width2, stage, number = 75, group = None, num_gcn=2, embedding_se=False):

        super(Framework_merge_base, self).__init__()
        self.embedding_se = embedding_se
        self.num_gcn = num_gcn
        self.width = width
        self.height = height
        self.width2 = width2
        self.height2 = height2
        self.num_classes = num_classes
        self.number = number
        self.stage = stage
        self.in_channel = in_channel
        self.in_channel2 = in_channel2

        self.bb = module = backbone()

        self.head_attr = group['head']
        self.arm_attr = group['arm']
        self.upper_attr = group['upper']
        self.lower_attr = group['lower']
        self.foot_attr = group['foot']
        self.features = module.features
        bias_use = False

        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=num_classes*10 , kernel_size=1, bias=bias_use)
        self.conv2 = nn.Conv2d(in_channels=self.in_channel2, out_channels=num_classes*10 , kernel_size=1, bias=bias_use)

        self.attention_linear1 = nn.Sequential(
            nn.Linear(in_features=height * width * 10, out_features=200),
            nn.ReLU()
        )

        self.attention_linear2 = nn.Sequential(
            nn.Linear(in_features=height2 * width2 * 10, out_features=200),
            nn.ReLU()
        )

        self.split_classify = nn.Linear(in_features=num_classes * self.width * self.height * 10, out_features=num_classes)
        self.down = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, inp, keypoints):

        x = self.features(inp)
        x2 = self.features[:self.stage](inp)

        embedding1 = self.conv(x).view(size=(-1, self.num_classes, self.width * self.height * 10))
        embedding = embedding1
        Pure_out1 = embedding.view(size=(-1, self.num_classes * self.width * self.height * 10))
        Pure_out1 = self.split_classify(Pure_out1)

        all_out = Pure_out1
        spatial_out = None
        hiera_out = None

        return all_out, spatial_out, hiera_out, None, Pure_out1, None



class Framework_merge_test4(nn.Module):

    def __init__(self, backbone, num_classes, in_channel, height, width, in_channel2, height2, width2, stage, number = 75, group = None, num_gcn=2, embedding_se=False):

        super(Framework_merge_test4, self).__init__()
        self.embedding_se = embedding_se
        self.num_gcn = num_gcn
        self.width = width
        self.height = height
        self.width2 = width2
        self.height2 = height2
        self.num_classes = num_classes
        self.number = number
        self.stage = stage

        self.bb = module = backbone()

        self.attr_index = {}
        self.attr_index['head'] = group['head']
        self.attr_index['arm'] = group['arm']
        self.attr_index['upper'] = group['upper']
        self.attr_index['lower'] = group['lower']
        self.attr_index['foot'] = group['foot']
        self.group_num = {}
        self.group_num['head'] = len(group['head'])
        self.group_num['arm'] = len(group['arm'])
        self.group_num['upper'] = len(group['upper'])
        self.group_num['lower'] = len(group['lower'])
        self.group_num['foot'] = len(group['foot'])
        self.features = module.features
        bias_use = False
        self.embedding = CoordConv2d_test(in_channels=in_channel, out_channels=num_classes * 10, kernel_size=1, num_classes = num_classes, 
                                   stride=1, bias=bias_use)
        
        self.embedding2 = CoordConv2d_test(in_channels=in_channel2, out_channels=num_classes * 10, kernel_size=1, num_classes = num_classes, 
                                   stride=1, bias=bias_use)
        self.mix_down = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        self.emb_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels=num_classes * 10, out_channels=num_classes * 10 // 16, kernel_size=1, bias=bias_use),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=num_classes * 10 // 16, out_channels=num_classes * 10, kernel_size=1, bias=bias_use),
            nn.ReLU(inplace=False),
            nn.Sigmoid()
        )
        
        self.attention_linear1 = nn.Sequential(
            nn.Linear(in_features=height * width * 10, out_features=200),
            nn.ReLU()
        )

        self.attention_linear2 = nn.Sequential(
            nn.Linear(in_features=height2 * width2 * 10, out_features=200),
            nn.ReLU()
        )


        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        print('componet all is loaded')
        self.all_gcn_linear = nn.ModuleList()
        for i in range(num_gcn):
            self.all_gcn_linear.append(
                nn.Sequential(nn.Linear(in_features=width * height * 10, out_features=width * height * 10),
                                nn.ReLU(inplace=True)
                                )
            )
        self.all_classify = nn.Linear(in_features=num_classes * width * height * 10, out_features=num_classes)

        self.classify_emb = nn.Linear(in_features=num_classes * width * height * 10, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.2)

        self.hiera_classify = nn.Linear(in_features=num_classes * width * height * 10, out_features=num_classes)

        self.embscale = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.gcnscale = nn.Parameter(torch.tensor(0.7), requires_grad=False)

        self.frontscale = nn.Parameter(torch.tensor(0.2), requires_grad=True)

        self.softmax = nn.Softmax(dim=2)

        self.direct_classify = nn.Linear(in_features=in_channel * self.height * self.width, out_features=num_classes)
        self.in_channel=in_channel

        self.split_classify = nn.Linear(in_features=self.num_classes * 400, out_features=num_classes)
        self.split_classify_ft = nn.Linear(in_features=self.num_classes * 200, out_features=num_classes)

        self.posescale = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.oriscale = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        self.mapper = KeyHeatModule_test2(num_classes, in_channel, width, height, self.group_num)
        self.mapper_front = KeyHeatModule_test2(num_classes, in_channel2, width2, height2, self.group_num)

        self.gap = 5
        self.gap_start = 10
        self.cross = True


    def forward_emb(self, embedding, mapper, attention_linear, embedding_linear, keypoints, width, height):

        embedding = embedding_linear(embedding, keypoints)
        
        embedding = self.dropout(embedding)
        
        embedding = torch.reshape(embedding, shape=(-1, self.num_classes, 10, width, height))

        head, arm, upper, lower, foot = mapper(embedding, keypoints)
        embedding = torch.reshape(embedding, shape=(-1, self.num_classes*10, width, height))

        
        embedding = embedding + embedding * self.emb_se(embedding)
        embedding = embedding.view(size=(-1, self.num_classes, 10, width, height))
        
        if self.group_num['head'] != 0:
            head = head.reshape(-1, self.group_num['head'], 1, width, height).float()
            embedding[:, self.attr_index['head']] = self.oriscale * embedding[:, self.attr_index['head']] + self.posescale * head * embedding[:, self.attr_index['head']]
        if self.group_num['arm'] != 0:
            arm = arm.reshape(-1, self.group_num['arm'], 1, width, height).float()
            embedding[:, self.attr_index['arm']] = self.oriscale * embedding[:, self.attr_index['arm']] + self.posescale * arm * embedding[:, self.attr_index['arm']]
        if self.group_num['upper'] != 0:
            upper = upper.reshape(-1, self.group_num['upper'], 1, width, height).float()
            embedding[:, self.attr_index['upper']] = self.oriscale * embedding[:, self.attr_index['upper']] + self.posescale * upper * embedding[:, self.attr_index['upper']]
        if self.group_num['lower'] != 0:
            lower = lower.reshape(-1, self.group_num['lower'], 1, width, height).float()
            embedding[:, self.attr_index['lower']] = self.oriscale * embedding[:, self.attr_index['lower']] + self.posescale * lower * embedding[:, self.attr_index['lower']]
        if self.group_num['foot'] != 0:
            foot = foot.reshape(-1, self.group_num['foot'], 1, width, height).float()
            embedding[:, self.attr_index['foot']] = self.oriscale * embedding[:, self.attr_index['foot']] + self.posescale * foot * embedding[:, self.attr_index['foot']]
        
        embedding = embedding.contiguous().view(size=(-1, self.num_classes, width * height * 10))

        feature = embedding.view(size=(-1, self.num_classes, 10, width,  height))

        embedding = attention_linear(embedding)
        embedding = self.dropout(embedding)

        return embedding, feature


    def forward(self, inp, keypoints):
        x = self.features(inp)
        x2 = self.features[:self.stage](inp)

        embedding1, _ = self.forward_emb(x, self.mapper, self.attention_linear1, self.embedding, keypoints, self.width, self.height)
        embedding2, _ = self.forward_emb(x2, self.mapper_front, self.attention_linear2, self.embedding2, keypoints, self.width2, self.height2)

        embedding = torch.cat([embedding1, embedding2], dim=-1)

        embedding = embedding.view(size=(-1, self.num_classes, 400))

        embedding_fc = embedding
        
        Pure_out1 = embedding_fc.view(size=(-1, self.num_classes * 400))

        Pure_out1 = self.split_classify(Pure_out1)
        
        all_out = Pure_out1
        spatial_out = None
        hiera_out = None
        return all_out, spatial_out, hiera_out, None, Pure_out1, None


    def feature(self, inp, keypoints):
        x = self.features(inp)
        x2 = self.features[:self.stage](inp)

        embedding1, feature1 = self.forward_emb(x, self.mapper, self.attention_linear1, self.embedding, keypoints, self.width, self.height)
        embedding2, feature2 = self.forward_emb(x2, self.mapper_front, self.attention_linear2, self.embedding2, keypoints, self.width2, self.height2)

        embedding = torch.cat([embedding1, embedding2], dim=-1)

        embedding = embedding.view(size=(-1, self.num_classes, 400))

        embedding_fc = embedding

        
        Pure_out1 = embedding_fc.view(size=(-1, self.num_classes * 400))
        Pure_out1 = self.split_classify(Pure_out1)
        
        all_out = Pure_out1
        spatial_out = None
        hiera_out = None
        return Pure_out1, feature1, feature2


if __name__ =='__main__':
    from module import backbone
    bb = backbone.Resnet50
    data = torch.randn(size=(4, 3, 224, 224))

