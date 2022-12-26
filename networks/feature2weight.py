import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from networks.layers import Conv3x3
from Camera import BackprojectDepth,Project3D
from utils import disp_to_depth,output2depth_use_scale
class wight_net(nn.Module):
    def __init__(self,args,num_ch_enc,B):
        super(wight_net, self).__init__()
        # 现在是
        #([B, 64, 192, 320])，([B, 64, 96, 160])，([B, 128, 48, 80])
        #三种分辨率的特征bilinear插值到192*320，然后直接grid sample
        self.depth2point = []
        self.point2rgb = []
        self.scales = [i for i in range(4,0,-1)]
        H,W = args.height,args.width
        self.H,self.W,self.B = H,W,B

        self.depth2point = BackprojectDepth(B, H//2, W//2)
        self.point2rgb = Project3D(B, H//2, W//2)
        c = num_ch_enc[0] + num_ch_enc[1] + num_ch_enc[2] + num_ch_enc[0]
        self.convs = torch.nn.Sequential(
            nn.Conv2d(c,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1)
        )

        self.fc = torch.nn.Sequential(
            nn.Linear(512*1*1,320),
            nn.ReLU(),
            nn.Linear(320,320),
            nn.ReLU(),
            nn.Linear(320,32)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self,args,inputs,outputs,refine_time_i):
        cameras = ['l','f']
        scale = 1 # 192x320
        for camera in cameras:
            opposite_camera = 'l' if camera=='f' else 'f'
            features_GS_name = 'features_f2l' if camera == 'l' else 'features_l2f'
            # features[0] = 192x320
            feature_0 = outputs['features',opposite_camera][0]
            feature_1 = outputs['features',opposite_camera][1]
            feature_2 = outputs['features',opposite_camera][2]

            feature = torch.cat([feature_0,
                F.interpolate(feature_1,[self.H//2, self.W//2],mode='bilinear', align_corners=False),
                F.interpolate(feature_2,[self.H//2, self.W//2],mode='bilinear', align_corners=False),
                                 ],dim=1)

            if refine_time_i==0:
                depth = output2depth_use_scale(args,inputs,outputs,camera,'init',0)
            elif refine_time_i == 1:
                depth = output2depth_use_scale(args,inputs,outputs,camera,'refine',0,refine_time=refine_time_i-1)#取上一次refine的结果做这次refine
            depth = F.interpolate(
                depth, [self.H // 2, self.W // 2], mode="bilinear", align_corners=False)

            cam_points = self.depth2point(depth,inputs['inv_K',camera,0, scale],
                                          inputs['extrinsics',camera,0].float())
            pix_coords = self.point2rgb(cam_points,inputs[('K',opposite_camera, 0, scale)],
                                        inputs['extrinsics_inv',opposite_camera,0].float())

            warp_feature = F.grid_sample(
                    feature, pix_coords,padding_mode='zeros',align_corners=True)
            feature_cat = torch.cat([outputs['features',camera][0],warp_feature],dim=1)

            x = self.convs(feature_cat)

            x = self.fc(x.view(x.shape[0],-1))
            x = x.unsqueeze(2).unsqueeze(3) + 1 / 32
            for i in range(3,-1,-1):
                disp_refine = outputs[('disp_code',camera,i)] * x
                disp_refine = torch.sum(disp_refine,dim=1).unsqueeze(1)
                outputs[('disp_refine_'+str(refine_time_i), camera, i)] = self.sigmoid(disp_refine)
        return outputs

if __name__=='__main__':
    pass
'''
input_features torch.Size([2, 64, 192, 320])
input_features torch.Size([2, 64, 96, 160])
input_features torch.Size([2, 128, 48, 80])
input_features torch.Size([2, 256, 24, 40])
input_features torch.Size([2, 512, 12, 20])
'''