import typing as t

import numpy as np
from pysam import VariantRecord

from splicelib.reference import Reference
from splicelib.utils import one_hot_encode, format_chromosome


PreprocessedRecord = t.NamedTuple('PreprocessedRecord', [
    ('gene', str), ('strand', str),
    ('ref_len', int), ('alt_len', int), ('del_len', int),
    ('ref_encoded', np.ndarray), ('alt_encoded', np.ndarray)
])


def preprocess(reference: Reference, dist_var: int, record: VariantRecord) \
        -> t.Tuple[t.List[PreprocessedRecord], t.Optional[str]]:
    """
    Preprocess a variant record.
    This function is heavily based on `get_delta_scores` from the original
    SpliceAI implementation. This is basically a minor refactoring of the
    preprocessing part of that function with some comments added to make it
    more readable.
    :param reference: a Reference object
    :param dist_var: maximum distance between the variant and gained/lost splice
    site
    :param record: a VariantRecord object
    :return: a list of preprocessed records ready for predictions with an
    optional logging message (a poor man's Writer monad for situations when you
    cannot afford the performance overhead of actual monads in Python).
    message
    """

    cov = 2 * dist_var + 1
    wid = 10000 + cov

    try:  # skip due to pysam formatting issues
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        return [], f'Bad variant record: {record}'

    chrom = format_chromosome(reference.long_chrom, record.chrom)
    feature_indices = reference.feature_indices(chrom, record.pos)
    if not feature_indices:
        return [], f'No overlapping features for variant record: {record}'
    # extract sequence from the reference; -1 in the left part of the slice
    # accounts for the fact that record.pos uses 1-based indexing, while Python
    # indexing is 0-based
    try:
        seq = reference.assembly[chrom][record.pos - wid // 2 - 1:record.pos + wid // 2].seq
    except (IndexError, ValueError):
        return [], f'Cannot extract sequence for variant record: {record}'

    # skip if the record reference allele doesn't match this segment in the
    # annotation sequence
    if seq[wid // 2:wid // 2 + len(record.ref)].upper() != record.ref:
        return [], f'Reference sequence does not match reference allele: {record}'

    if len(seq) != wid:
        return [], f'The variant is too close to the chromosome end: {record}'

    if len(record.ref) > 2 * dist_var:
        return [],  f'The reference allele is too long: {record}'

    preprocessed_records = []
    # loop through all combinations of alternate alleles and feature indices
    for idx in feature_indices:
        gene = reference.genes[idx]
        strand = reference.strands[idx]
        for alt in record.alts:
            # skip missing alternate alleles
            if '.' in alt or '-' in alt or '*' in alt:
                continue
            if '<' in alt or '>' in alt:
                continue
            if len(record.ref) > 1 and len(alt) > 1:
                continue
            # get distance to transcript and exon boundaries
            d_tx_start, d_tx_end, d_exon_boundary = reference.feature_distances(idx, record.pos)
            # use padding if the window goes outside of gene boundaries
            pad_size = [max(wid // 2 + d_tx_start, 0), max(wid // 2 - d_tx_end, 0)]
            ref_len = len(record.ref)  # reference allele length
            alt_len = len(alt)  # alternate allele length
            del_len = max(ref_len - alt_len, 0)  # deletion length

            # create a padded version of reference and alternative sequence
            ref_pad = 'N' * pad_size[0] + seq[pad_size[0]:wid - pad_size[1]] + 'N' * pad_size[1]
            # do the same for the alternate allele by inserting the alternative
            # allele into the reference sequence
            alt_pad = ref_pad[:wid // 2] + str(alt) + ref_pad[wid // 2 + ref_len:]

            # one-hot encode the sequences (size=(wid, 4))
            ref_encoded = one_hot_encode(ref_pad)
            alt_encoded = one_hot_encode(alt_pad)

            # reverse-complement encoded sequences if the strand is negative
            # (see documentation on `one_hot_encode` to understand why this
            #  works)
            if strand == '-':
                ref_encoded = ref_encoded[::-1, ::-1]
                alt_encoded = alt_encoded[::-1, ::-1]
            preprocessed_record = PreprocessedRecord(
                gene, strand, ref_len, alt_len, del_len, ref_encoded, alt_encoded
            )
            preprocessed_records.append(preprocessed_record)
    return (
        (preprocessed_records, None) if preprocessed_records else
        [], f'No valid alternative alleles for record: {record}'
    )


if __name__ == '__main__':
    raise RuntimeError
