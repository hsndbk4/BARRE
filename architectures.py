
from archs.mobilenet import MobileNetV1
from archs.resnet18 import ResNet18
from archs.resnet20 import resnet20
from datasets import get_num_classes

def get_architecture(args):
    num_classes = get_num_classes(args)
    if args.model == 'res18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'res20':
        model = resnet20(num_classes=num_classes)
    elif args.model == 'mbv1':
        model = MobileNetV1(num_classes=num_classes)
    else:
        raise ValueError('Unsupported model ' + args.model)
    return model
