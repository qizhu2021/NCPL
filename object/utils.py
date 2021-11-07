import network


def get_model(net, num_classes):
    if net[0:3] == 'res':
        return network.ResBase(net, num_classes).cuda()
    elif net[0:3] == 'vgg':
        return network.VGGBase(net, num_classes).cuda()
    elif net[0:3] == 'inc':
        return network.InceptionBase(num_classes).cuda()
    elif net[0:3] == 'goo':
        return network.GoogLeNet(num_classes).cuda()
    elif net[0:3] == 'ale':
        return network.AlexNet(num_classes).cuda()


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer