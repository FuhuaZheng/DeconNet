import torch 
import torch.nn as nn 
# from torchvision.transforms import Resize
import torch.nn.functional as F


class Conv_firstloop_Block(nn.Module):
    def __init__(self, in_c, out_c, padding = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d((padding, padding, 0, 0)), nn.Conv2d(in_c, out_c, 5, 1, 1, padding_mode="reflect", bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.ReflectionPad2d((padding, padding, 0, 0)), nn.Conv2d(out_c, out_c, 5, 1, 1, padding_mode="reflect", bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.ReflectionPad2d((padding, padding, 0, 0)), nn.Conv2d(out_c, out_c, 5, 1, 1, padding_mode="reflect", bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
        )
        
    def forward(self, x):
        padded_x = torch.cat([x[:, :, -3:, :], x, x[:, :, :3 :]], dim=2)
        output = self.layers(padded_x)
        return output
    
class Conv_lastloop_Block(nn.Module):
    def __init__(self, in_c, out_c, padding = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 0, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 0, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 0, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, padding_mode="reflect", bias=False),nn.BatchNorm2d(out_c),nn.ReLU(),
        )
        
    def forward(self, x):
        padded_x = torch.cat([x[:, :, -2:, :], x, x[:, :, :2 :]], dim=2)
        output = self.layers(padded_x)
        return output
    
class Conv_2loop_Block(nn.Module):
    def __init__(self, in_c, out_c, padding = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d((padding, padding, 0, 0)), nn.Conv2d(in_c, out_c, 3, 1, 0, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.ReflectionPad2d((padding, padding, 0, 0)), nn.Conv2d(out_c, out_c, 3, 1, 0, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, padding_mode="reflect", bias=False),nn.BatchNorm2d(out_c),nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, padding_mode="reflect", bias=False),nn.BatchNorm2d(out_c),nn.ReLU(),
        )
        
    def forward(self, x):
        padded_x = torch.cat([x[:, :, -2:, :], x, x[:, :, :2 :]], dim=2)
        output = self.layers(padded_x)
        return output
    

class Conv1_Block(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c, (1,3), 1, (0,1), padding_mode="reflect",bias=False),nn.BatchNorm2d(in_c),nn.ReLU(),
            nn.Conv2d(in_c, out_c, (1,3), 1, (0,1), padding_mode="reflect",bias=False),nn.BatchNorm2d(out_c),nn.ReLU(),
            nn.Conv2d(out_c, out_c, (1,3), 1, (0,1), padding_mode="reflect",bias=False),nn.BatchNorm2d(out_c),nn.ReLU(),
            nn.Conv2d(out_c, out_c, (1,3), 1, (0,1), padding_mode="reflect",bias=False),nn.BatchNorm2d(out_c),nn.ReLU(),
        )

    def forward(self, x):
        output = self.layers(x)
        return output
    

class Down1_Sample_Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

    def forward(self, x):
        return self.layers(x)
    
    
class Down2_Sample_Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2,2)
        ) 

    def forward(self, x):
        return self.layers(x)
    
    
class Up1_Sample_Block(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c//2, (1,3), 1, (0,1), padding_mode="reflect",bias=False),nn.ReLU(),
        )
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2) 
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self,x,feature):
        x = self.layers(x)
        x = torch.nn.functional.interpolate(input=x,scale_factor=(1,2), mode="nearest")
        res = torch.cat((x,feature),dim=1)
        return res
    
    
class Up2_Sample_Block(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c//2, 3, 1, padding=1, padding_mode="reflect", bias=False),nn.ReLU(),
        )

    def forward(self,x,feature):
        x = self.layers(x)
        x = torch.nn.functional.interpolate(input=x,scale_factor=2, mode="nearest")
        res = torch.cat((x,feature),dim=1)
        return res


class Output(nn.Module):
    def __init__(self,in_c, out_c) -> None:
        super().__init__()
        self.layers = self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, (1,3), 1, (0,1), bias=False),nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = Down1_Sample_Block()
        self.down2 = Down2_Sample_Block()

        self.conv20 = Conv_firstloop_Block(2,4)

        self.conv111 = Conv1_Block(4,8)
        self.conv121 = Conv1_Block(8,16)
        self.conv131 = Conv1_Block(16,32)
        
        self.conv21 = Conv_2loop_Block(32,64)
        self.conv22 = Conv_2loop_Block(64,128)
        self.conv23 = Conv_2loop_Block(128,256)       
        self.conv24 = Conv_2loop_Block(256,512)

        self.conv232 = Conv_2loop_Block(256,128)
        self.conv222 = Conv_2loop_Block(128,64)
        self.conv212 = Conv_2loop_Block(64,32)

        self.conv122 = Conv1_Block(32,16)
        self.conv112 = Conv1_Block(16,8)
        self.conv102 = Conv1_Block(8,4)


        # self.conv202 = Conv_lastloop_Block(8,4)
        
        self.up23 = Up2_Sample_Block(256)
        self.up22 = Up2_Sample_Block(128)
        self.up21 = Up2_Sample_Block(64)

        self.up12 = Up1_Sample_Block(32)
        self.up11 = Up1_Sample_Block(16)
        self.up10 = Up1_Sample_Block(8)
        self.drop = nn.Dropout(0.05)
        
        self.out = Output(4,1)

    def forward(self,x):
        
        out20 = self.conv20(x)
        out11 = self.conv111(self.down1(out20))
        out12 = self.conv121(self.down1(out11))
        out13 = self.conv131(self.down1(out12))

        out21 = self.conv21(self.down2(out13))
        out22 = self.conv22(self.down2(out21))
        out23 = self.conv23(self.down2(out22))

        out232 = self.conv232(self.up23(out23, out22))
        out222 = self.conv222(self.up22(out232, out21)) # out222 size: torch.Size([1, 64, 32, 32])
        out212 = self.conv212(self.up21(out222, out13)) # out212 size: torch.Size([1, 32, 64, 64])

        out122 = self.conv122(self.up12(out212, out12)) # out122 size: torch.Size([1, 16, 64, 256])
        out112 = self.conv112(self.up11(out122, out11)) # out112 size: torch.Size([1, 8, 64, 512])
        out102 = self.conv102(self.up10(out112, out20)) # out202 size: torch.Size([1, 4, 64, 1024])
        x = self.drop(out102)
        out = self.out(x) # output_size: torch.Size([1, 1, 64, 1024])
        return out
        
def save_checkpoint(model, weightfile):
    if torch.cuda.device_count() > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({
        'model_state_dict': model_state_dict,
    }, weightfile)

# if __name__ == "__main__":
#     x = torch.randn((32,2,64,512))
#     print("input_size:",x.shape)
#     net = UNet()
#     y = net(x)
#     print("output_size:",y.shape)
#     outputs = y.squeeze(dim=1)
#     print(f'squeezed shape:{outputs.shape}')
    
    # torch.onnx.export(net, y, './onnx_model.onnx', opset_version=11)