import torch


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator
    )


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Divide the numerator by the denominator and check that there is no remainder."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: The tensor to split.
        num_partitions: The number of partitions to split the tensor into.
        contiguous_split_chunks: Whether to return contiguous split chunks.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide_and_check_no_remainder(
        tensor.size()[last_dim], num_partitions
    )
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


class VocabUtility:
    """
    Split the vocabulary into `world_size` chunks and return the first and last
    index of the vocabulary belonging to the `rank` partition.

    Note that indices in [first, last]
    """

    @staticmethod
    def get_vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank: int, _: int
    ) -> tuple[int, int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> tuple[int, int]:
        per_partition_vocab_size = divide_and_check_no_remainder(
            global_vocab_size, world_size
        )
        return VocabUtility.get_vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
