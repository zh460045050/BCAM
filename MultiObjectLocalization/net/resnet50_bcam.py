import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import torch

class Net(nn.Module):

    def __init__(self, num_head):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        #self.classifier = nn.Conv2d(2048, 20, 1, bias=False

        self.fc_attp_fore = nn.Conv2d(2048, num_head, 1, bias=False)
        self.fc_attp_back = nn.Conv2d(2048, num_head, 1, bias=False)

        self.fc8_fore = nn.Conv2d(2048, 20, 1, bias=False)
        self.fc8_back = nn.Conv2d(2048, 20, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8_fore.weight)
        torch.nn.init.xavier_uniform_(self.fc8_back.weight)
        torch.nn.init.xavier_uniform_(self.fc_attp_fore.weight)
        torch.nn.init.xavier_uniform_(self.fc8_back.weight)


        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.fc8_fore, self.fc8_back, self.fc_attp_fore, self.fc_attp_back, ])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)
        
        f_pixel = x.clone()

        att_f = self.fc_attp_fore(x)
        b, c, h, w = att_f.shape
        att_f = att_f.view(b, c, h*w)
        att_f = F.softmax(att_f ,dim=2)

        att_b = self.fc_attp_back(x)
        b, c, h, w = att_b.shape
        att_b = att_b.view(b, c, h*w)
        att_b = F.softmax(att_b ,dim=2)


        #x = torchutils.gap2d(x, keepdims=True)
        #x = self.classifier(x)
        b, c, h, w = f_pixel.shape
        f_image_fore = torch.bmm(att_f, f_pixel.view(b, c, h*w).permute(0, 2, 1)).mean(dim=1).view(b, c, 1, 1)
        f_image_back = torch.bmm(att_b, f_pixel.view(b, c, h*w).permute(0, 2, 1)).mean(dim=1).view(b, c, 1, 1)

        label_fore = self.fc8_fore(f_image_fore).view(-1, 20)
        label_back = self.fc8_back(f_image_back).view(-1, 20)

        label_back_rev = self.fc8_fore(f_image_back).view(-1, 20)
        label_fore_rev = self.fc8_back(f_image_fore).view(-1, 20)

        return {"logits":label_fore, "logits_fore_rev":label_fore_rev, "logits_back":label_back, "logits_back_rev":label_back_rev}

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, num_head):
        super(CAM, self).__init__(num_head)

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        #x = F.conv2d(x, self.classifier.weight)
        x = self.fc8_fore(x)
        x = F.relu(x)

        #cam_fore = self.fc8_fore(x)
        #cam_back = self.fc8_back( f_pixel)

        x = x[0] + x[1].flip(-1)

        return x


