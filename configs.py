import argparse


def getConfig():
    parser = argparse.ArgumentParser()


    parser.add_argument('--model_name', 'm', default='unet', help='default : unet, [unet, fpn]')
    parser.add_argument('--k_folds', 'k', default=1, type=int, help='default : 1')
    parser.add_argument('--BACKBONE', 'backbone', default='resnet34', help='default : resnet34, [  ]')
    parser.add_argument('--PRETRAINED', default=None, help='default : None')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='default : 150, number of total epochs to run')
    parser.add_argument('--BATCH_SIZE', default=32, type=int, help='default : 32, mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--WORKERS', default=0, type=int, help='default : 0, WORKERS')
    parser.add_argument('--Threshold', default=0.5, type=float, help='default : 0.5')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    config = getConfig()
    config = vars(config)
    print(config)