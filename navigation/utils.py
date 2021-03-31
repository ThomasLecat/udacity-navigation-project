import torch


class OneHot:
    """
    Convert an integer LongTensor of size `batch_size` into a one-hot tensor.
    """

    def __init__(self, batch_size: int, num_digits: int, device):
        self.one_hot = torch.FloatTensor(batch_size, num_digits).to(device)

    def __call__(self, labels):
        self.one_hot.zero_()
        return self.one_hot.scatter_(1, labels.view(-1, 1), 1)
