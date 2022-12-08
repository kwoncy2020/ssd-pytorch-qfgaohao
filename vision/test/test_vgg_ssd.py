from ..ssd.vgg_ssd import create_vgg_ssd
import itertools
import torch
import tempfile


def test_create_vgg_ssd():
    for num_classes in [2, 10, 21, 100]:
        _ = create_vgg_ssd(num_classes)


def test_forward():
    for num_classes in [2]:
        net = create_vgg_ssd(num_classes)
        net.init()
        net.eval()
        x = torch.randn(2, 3, 300, 300)
        confidences, locations = net.forward(x)
        assert confidences.size() == torch.Size([2, 8732, num_classes])
        assert locations.size() == torch.Size([2, 8732, 4])
        assert confidences.nonzero().size(0) != 0
        assert locations.nonzero().size(0) != 0


def test_save_model():
    net = create_vgg_ssd(10)
    net.init()
    # params = [
    #         {'params': net.base_net.parameters(), 'lr': 1e-3},
    #         {'params': itertools.chain(
    #             net.source_layer_add_ons.parameters(),
    #             net.extras.parameters()
    #         ), 'lr': 1e-3},
    #         {'params': itertools.chain(
    #             net.regression_headers.parameters(),
    #             net.classification_headers.parameters()
    #         )}
    #     ]
    # optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9,
    #                             weight_decay=5e-4)
    with tempfile.TemporaryFile() as f:
        net.save(f)


def test_save_load_model_consistency():
    net = create_vgg_ssd(20)
    net.init()
    model_path = tempfile.NamedTemporaryFile().name
    # params = [
    #         {'params': net.base_net.parameters(), 'lr': 1e-3},
    #         {'params': itertools.chain(
    #             net.source_layer_add_ons.parameters(),
    #             net.extras.parameters()
    #         ), 'lr': 1e-3},
    #         {'params': itertools.chain(
    #             net.regression_headers.parameters(),
    #             net.classification_headers.parameters()
    #         )}
    #     ]
    # optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9,
    #                             weight_decay=5e-4)
    net.save(model_path)
    net_copy = create_vgg_ssd(20)
    net_copy.load(model_path)

    net.eval()
    net_copy.eval()

    for _ in range(1):
        x = torch.randn(1, 3, 300, 300)
        confidences1, locations1 = net.forward(x)
        confidences2, locations2 = net_copy.forward(x)
        assert (confidences1 == confidences2).long().sum() == confidences2.numel()
        assert (locations1 == locations2).long().sum() == locations2.numel()
