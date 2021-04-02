import torch

from navigation.replay_buffer import SampleBatch, TorchSampleBatch


class OneHot:
    """
    Convert an integer LongTensor of size `batch_size` into a one-hot tensor.
    """

    def __init__(self, batch_size: int, num_digits: int, device):
        self.one_hot = torch.FloatTensor(batch_size, num_digits).to(device)

    def __call__(self, labels: torch.LongTensor) -> torch.FloatTensor:
        self.one_hot.zero_()
        return self.one_hot.scatter_(1, labels.view(-1, 1), 1).clone()


def convert_to_torch(sample_batch: SampleBatch, device) -> TorchSampleBatch:
    return TorchSampleBatch(
        **{
            key: torch.from_numpy(value).to(device)
            for key, value in sample_batch._asdict().items()
        }
    )
