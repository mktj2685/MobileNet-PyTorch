import torch
import torch.nn as nn


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup):
            # Pytorch does not support same padding when stride > 1.
            # see https://github.com/pytorch/pytorch/issues/3867
            dilation = 1
            kernel_size = 3
            stride = 2
            padding = max(0, dilation * (kernel_size - 1) - (inp - 1) % stride)
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, 2, padding),
                nn.BatchNorm2d(num_features=oup),
                nn.ReLU()
            )

        def conv_dw_sep(inp, oup, Dk, s):
            # Pytorch does not support same padding when stride > 1.
            # see https://github.com/pytorch/pytorch/issues/3867
            dilation = 1
            padding = max(0, dilation * (Dk - 1) - (inp - 1) % s)
            return nn.Sequential(
                # conv dw
                # see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                nn.Conv2d(inp, inp, Dk, s, groups=inp, padding=padding),
                nn.BatchNorm2d(inp),
                nn.ReLU(),

                # conv pw
                nn.Conv2d(inp, oup, 1, 1),
                nn.BatchNorm2d(oup),
                nn.ReLU()
            )

        self.layers = nn.Sequential(
            conv_bn(3, 32),

            conv_dw_sep(32, 64, 3, 1),
            conv_dw_sep(64, 128, 3, 2),
            conv_dw_sep(128, 128, 3, 1),
            conv_dw_sep(128, 256, 3, 2),
            conv_dw_sep(256, 256, 3, 1),
            conv_dw_sep(256, 512, 3, 2),
            *[conv_dw_sep(512, 512, 3, 1) for _ in range(5)],
            conv_dw_sep(512, 1024, 3, 2),
            conv_dw_sep(1024, 1024, 3, 2),

            nn.AvgPool2d(7)
        )

        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)  # B x C x H x W
    model = MobileNetV1()
    output = model(x)
    print(output.shape)
