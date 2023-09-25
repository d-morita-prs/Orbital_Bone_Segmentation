import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from glob import glob
import os
import SimpleITK as sitk
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d( in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.2)
    )

  def forward(self, x):
    return self.block(x)

class DownBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.down_block = nn.Sequential(
        nn.MaxPool2d(2, stride=2),
        ConvBlock(in_ch, out_ch),
    )

  def forward(self, x):
    return self.down_block(x)

class UpBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    self.conv = ConvBlock(in_ch, out_ch)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    return self.conv(torch.cat((x1, x2), dim=1))

class UNet6(nn.Module):
  def __init__(self, n_class):
    super().__init__()
    self.n_class = n_class

    self.input_block = ConvBlock(1, 64)

    self.down_block_1st = DownBlock( 64,  128)
    self.down_block_2nd = DownBlock(128,  256)
    self.down_block_3rd = DownBlock(256,  512)
    self.down_block_4th = DownBlock(512, 1024)
    self.down_block_5th = DownBlock(1024, 2048)

    self.up_block_1st = UpBlock(2048, 1024)
    self.up_block_2nd = UpBlock(1024, 512)
    self.up_block_3rd = UpBlock( 512, 256)
    self.up_block_4th = UpBlock( 256, 128)
    self.up_block_5th = UpBlock( 128,  64)

    self.output_block = nn.Sequential(
        nn.Conv2d(64, self.n_class, 1)
    )

  def forward(self, x):
    skip1 = y = self.input_block(x)
    skip2 = y = self.down_block_1st(y)
    skip3 = y = self.down_block_2nd(y)
    skip4 = y = self.down_block_3rd(y)
    skip5 = y = self.down_block_4th(y)
    skip6 = y = self.down_block_5th(y)

    y = self.up_block_1st(y, skip5)
    y = self.up_block_2nd(y, skip4)
    y = self.up_block_3rd(y, skip3)
    y = self.up_block_4th(y, skip2)
    y = self.up_block_5th(y, skip1)

    return self.output_block(y)

def visualize_mhd(model,device,mhd,cwd):
    
    mhd_path=Path(mhd)
    ct_volume=sitk.ReadImage(mhd_path.as_posix())
    orig_spacing = ct_volume.GetSpacing()
    orig_origin = ct_volume.GetOrigin()
    ct_slices=sitk.GetArrayFromImage(ct_volume)
    
    model.eval()
    with torch.no_grad():
        
        output_slices=[]
        
        for j in range(ct_slices.shape[0]):
            input_slice=torch.tensor(ct_slices[j][np.newaxis][np.newaxis]).to(device)
            output=model(input_slice.float())
            softmax_output=F.softmax(output,dim=1)
            arg_soft_output=torch.argmax(softmax_output,dim=1)
            arg_soft_output=arg_soft_output.cpu().detach().numpy().copy()
            arg_soft_output=np.squeeze(arg_soft_output,axis=0)
            output_slices.append(arg_soft_output)
        
        output_slices=np.array(output_slices)
        
        label_pred=sitk.GetImageFromArray(output_slices)
        label_pred.SetSpacing(orig_spacing)
        label_pred.SetOrigin(orig_origin)
        sitk.WriteImage(label_pred,f"{cwd}/output/{mhd_path.stem}_label.mhd") 

model=UNet6(n_class=3)
model.load_state_dict(torch.load("./weight_115cpu.pth"))
model=model.to(device)

cwd=os.getcwd()
mhd=np.array(sorted(glob(os.path.join(cwd,"images","*.mhd"))))

for i in mhd:
  visualize_mhd(model,device,i,cwd)
