import torch

def main():
    x = torch.rand(512, 1, 96, 96).cuda()
    w = torch.rand(1, 1, 50, 50).cuda()
    y = torch.nn.functional.conv2d(x, w)


if __name__ == '__main__':
    main()
