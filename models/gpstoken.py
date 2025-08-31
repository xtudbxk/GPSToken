import torch
import torch.nn as nn
import torchvision

from models.attnblock import AttnBlock
from models.vqvae import ResnetBlock, Decoder
from gscuda.gswrapper import gaussiansplatting_render

class GPSToken(nn.Module):
    def __init__(self, gpsconfig, decoderconfig):
        super().__init__()
        self.gpsconfig = gpsconfig
        self.decoderconfig = decoderconfig
        
        # decoder
        self.decoder = Decoder(**decoderconfig)

        # the conditional encoder
        self.preprocess_img = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, padding=1),
                ResnetBlock(in_channels=128),
                ResnetBlock(in_channels=128))

        rois = gpsconfig["rois"] 
        self.roialign = torchvision.ops.RoIAlign(output_size=(rois,rois), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
        self.roialign_layer = nn.Conv2d(128, gpsconfig["gpsenc_c"], kernel_size=1, stride=1)

        # init the gs encoder
        self.gps_embed = nn.Embedding(1, gpsconfig["gpsenc_c"]) # init gs embed
        self.attnblocks = nn.ModuleList([AttnBlock(gpsconfig["gpsenc_c"], roic=rois**2) for _ in range(gpsconfig["gpsenc_n"])])

        # gs coords
        self.gps_embed_layer = nn.Sequential(
                nn.Linear(5, gpsconfig["gpsenc_c"]),
                nn.GELU(),
                nn.Linear(gpsconfig["gpsenc_c"], gpsconfig["gpsenc_c"]))

        # to gs parameters
        self.to_gps = nn.ModuleList([
            nn.Linear(gpsconfig["gpsenc_c"], 5), # [sigma_x,sigma_y,rho,x,y]
            nn.Linear(gpsconfig["gpsenc_c"], gpsconfig["gps_c"]), # features
            ])
        self.to_gps[0].weight.data.fill_(0.0)
        self.to_gps[0].bias.data.fill_(0.0)

    def render_gpstoken(self, gpstoken, size=(64,64), dmax=10):
        rendered_result = []
        for bi in range(gpstoken.shape[0]):
            sigmas = gpstoken[bi, :, :3]
            coords = gpstoken[bi, :, 3:5]
            colors = gpstoken[bi, :, 5:]
            rendered_result.append(gaussiansplatting_render(sigmas, coords, colors, size, dmax).permute(2, 0, 1))
        return torch.stack(rendered_result)

    def encode(self, x, init_gpscodes=None, regions=None):

        # condition
        x_preprocess = self.preprocess_img(x) 

        # roi condition
        with torch.no_grad():
            _b, _, _h, _w = x.shape
            batch_index = torch.Tensor([_ for _ in range(_b)]).to(torch.float32).to(x.device)
            batch_index = batch_index.view(-1,1,1).repeat(1,self.gpsconfig["gps_num"],1)
            regions[:,:,0] = (_w-1) * (regions[:,:,0]+1)/2.0
            regions[:,:,1] = (_h-1) * (regions[:,:,1]+1)/2.0
            regions[:,:,2] = (_w-1) * (regions[:,:,2]+1)/2.0
            regions[:,:,3] = (_h-1) * (regions[:,:,3]+1)/2.0
            roi = torch.cat([batch_index, regions], dim=-1).view(-1, 5)

        roi_conds = self.roialign(x_preprocess, roi) # [b*gs_n, c, 9, 9)
        roi_conds = self.roialign_layer(roi_conds).flatten(2) # [b, c, 81]

        # initialize 
        init_gpscodes_embed = self.gps_embed_layer(init_gpscodes[:,:,:5]) #[sigma_x,sigma_y,rho,x,y]
        gpsembed = self.gps_embed.weight[None, :, :]
        gpsembed = gpsembed.repeat(x.shape[0], self.gpsconfig["gps_num"], 1)
        gpsembed = gpsembed + init_gpscodes_embed

        # gps encoder
        for mi in range(len(self.attnblocks)):
            gpsembed = self.attnblocks[mi](gpsembed, roi_conds)

        # obtain current gs code
        gps_delta = self.to_gps[0](gpsembed)

        _inverse_sigmoid_gpscode = -torch.log(1.0/init_gpscodes[:,:,:2] - 1.0) # [sigma_x, sigma_y]
        _inverse_2sigmoid1_gpscode = -torch.log(2.0/(1.0+init_gpscodes[:,:,2:5]) - 1.0) # [rho, x, y]
        gps_parameters = torch.cat([
            1.0/(1.0+torch.exp(-(_inverse_sigmoid_gpscode + gps_delta[:,:,:2]))),
            2.0/(1.0+torch.exp(-(_inverse_2sigmoid1_gpscode + gps_delta[:,:,2:5])))-1.0,
            ], dim=-1)

        contents = self.to_gps[-1](gpsembed)
        gpscodes = torch.cat([gps_parameters, contents], dim=-1)

        return gpscodes

    def decode(self, gpstoken):
        rendered = self.render_gpstoken(gpstoken, size=self.gpsconfig["gps_rs"], dmax=self.gpsconfig["gps_dmax"])
        dec = self.decoder(rendered)
        return dec

    def forward(self, x, init_gpscodes=None, regions=None):
        gpstoken = self.encode(x, init_gpscodes=init_gpscodes, regions=regions)
        recon = self.decode(gpstoken)
        return recon
