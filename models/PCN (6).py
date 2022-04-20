import torch
import torch.nn as nn
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2

class STN(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels):
        # RGB : Nx3 (3 -> x, y, z coordinates) & N : number of points.
        super(STN, self).__init__()

        out_channels=in_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2_1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.conv2_2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.conv2_3 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.conv2_4 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        #self.mp1 = torch.nn.MaxPool1d((2))
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        #print('stn in',x.size())                
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        x= x1 + x2 + x3 + x4
        #x=torch.nn.functional.interpolate(x, (64))
        #print('stn out',x.size())
        return x

class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """
    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y

@MODELS.register_module()
class PCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.number_fine = config.num_pred
        self.encoder_channel = config.encoder_channel
        self.encoder_channel_3 = 3072
        grid_size = 4 # set default
        self.stn = STN(256)
        self.csam = CSAM(256)

        self.stn1 = STN(1024)
        self.csam1 = CSAM(1024)

        self.grid_size = grid_size
        assert self.number_fine % grid_size**2 == 0
        self.number_coarse = self.number_fine // (grid_size ** 2 )
        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )

        self.third_conv = nn.Sequential(
            nn.Conv1d(2048,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )

        # self.mlp3 = nn.Sequential(
        #   nn.Linear(2048,1024),
        #   nn.ReLU(),
        # )
        ## 
        #self.mlp1 = nn.Sequential(
        #    nn.Linear(self.encoder_channel,1024),
        #    nn.ReLU(),
        #    nn.Linear(1024,1024),
        #    nn.ReLU(),
        #    nn.Linear(1024,3*self.number_coarse)
        #)
        ## 
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(self.encoder_channel,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,3*self.number_coarse)
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(self.encoder_channel,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,3*self.number_coarse)
        # )
        # 
        # self.mlp3 = nn.Sequential(
        #     nn.Linear(self.encoder_channel_3,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,3*self.number_coarse)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_channel,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,3*self.number_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024+3+2,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512,3,1)
        )
        a = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2).cuda() # 1 2 S
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        bs , n , _ = xyz.shape
        # encoder

        #print(xyz.transpose(2,1).shape)
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        # print("feature", feature.shape)
        # feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        # print("feature_global", feature_global.shape)
        
        feature_global1 = self.stn(feature)
        # print("feature_global1", feature_global1.shape)

        feature_global2 = self.csam(feature)
        # print("feature_global2", feature_global2.shape)


        feature = torch.cat([feature_global1.expand(-1,-1,n), feature_global2.expand(-1,-1,n)], dim=1)# B 512 n
        # print("feature", feature.shape)


        # exit()
        
        feature = torch.max(feature,dim=2,keepdim=True)[0]

        
        # print(feature.shape)
        # feature = self.mlp2(feature) # B 1024 n
        # feature_global = torch.max(feature,dim=2,keepdim=True)[0] # B 1024

        feature = self.second_conv(feature) # B 1024 n
        feature_global1 = self.stn1(feature)
        feature_global2 = self.csam1(feature)
        feature_global =  torch.cat([feature_global1.expand(-1,-1,n), feature_global2.expand(-1,-1,n)], dim=1) #feature_global[..., 0]     # (B x 1024 x 1) -> (B x 1024)

        feature = torch.max(feature_global,dim=2,keepdim=True)[0]

        # print("feature 2", feature.shape)

        # 
        # feature = self.mlp3(feature) # B 1024 n
        # feature_global = torch.max(feature,dim=2,keepdim=True)[0] # B 1024
        # feature_global1 = self.stn1(feature_global)
        # feature_global2 = self.csam1(feature_global)
        # feature_global =  torch.cat([feature_global1.expand(-1,-1,n), feature_global2.expand(-1,-1,n)], dim=1) #feature_global[..., 0]     # (B x 1024 x 1) -> (B x 1024)
        # feature_global = torch.max(feature,dim=2,keepdim=True)[0]

        # print(feature_global.shape)

        feature = self.third_conv(feature)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]

        # print('feature_3', feature_global.shape)
        # exit()



        # decoder
        coarse = self.mlp(feature_global).reshape(-1,self.number_coarse,3) # B M 3
        print('coarse', coarse.shape)
        point_feat=coarse.transpose(2,1)
        print('point_feat', coarse.shape)
        point_feat = coarse.unsqueeze(2).expand(-1,-1,self.grid_size**2,-1) # B M S 3
        point_feat = point_feat.reshape(-1,self.number_fine,3).transpose(2,1) # B 3 N
        
        seed = self.folding_seed.unsqueeze(2).expand(bs,-1,self.number_coarse, -1) # B 2 M S
        seed = seed.reshape(bs,-1,self.number_fine)  # B 2 N

        # print(feature_global.shape)
        # print()

        feature_global = feature_global.unsqueeze(2).expand(-1,-1,self.number_fine) # B 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # B C N
    
        fine = self.final_conv(feat) + point_feat   # B 3 N

        return (coarse.contiguous(), fine.transpose(1,2).contiguous())