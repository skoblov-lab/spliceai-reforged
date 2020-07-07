import numpy as np

BASE_TRANSLATION = str.maketrans('NACGT', '\x00\x01\x02\x03\x04')
BASE_MAP = np.asarray([[0, 0, 0, 0],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])


def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot-encode a nucleotide sequence. The mappings for N, A, C, G and T
    (in this precise order) are shown in variable BASE_MAP. Given an encoded
    sequence `x`, `x[:, ::-1, ::-1]` produces the encoding for its
    reverse-complement.
    :param seq: a nucleotide sequence; supported characters (case-insensitive):
    N, A, C, G, T
    :return: an encoded sequence
    """
    translated = seq.upper().translate(BASE_TRANSLATION).encode()
    return BASE_MAP[np.frombuffer(translated, np.int8)]


def format_chromosome(long: bool, chrom: str) -> str:
    """
    Ensure a chromosome identifier is formatted the right way (long or short)
    :param long: if True, enforce long formatting, e.g. 'chr1' instead of '1'
    :param chrom: a chromosome identifier
    :return:
    """
    return (
        chrom.lstrip('chr') if not long else
        chrom if chrom.startswith('chr') and long else
        'chr' + chrom
    )


if __name__ == '__main__':
    raise RuntimeError
