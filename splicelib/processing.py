import typing as t
import operator as op
from itertools import chain, groupby
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from pysam import VariantRecord

from tensorflow.keras.models import Model
from splicelib.reference import Reference
from splicelib.utils import one_hot_encode, format_chromosome


PreprocessedAllele = t.NamedTuple('PreprocessedAllele', [
    ('ref', str), ('alt', str), ('gene', str), ('strand', str),
    ('d_exon_boundary', int), ('x_ref', np.ndarray), ('x_alt', np.ndarray)
])


def preprocess(reference: Reference, dist_var: int, record: VariantRecord) \
        -> t.Tuple[t.List[PreprocessedAllele], t.Optional[str]]:
    """
    Preprocess a variant record.
    This function is heavily based on `get_delta_scores` from the original
    SpliceAI implementation. This is basically a minor refactoring of the
    pre-processing part of that function with some comments added to make it
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
            len_ref = len(record.ref)  # reference allele length

            # create a padded version of reference and alternative sequence
            ref_pad = 'N' * pad_size[0] + seq[pad_size[0]:wid - pad_size[1]] + 'N' * pad_size[1]
            # cut out the reference allele and insert the alternative allele
            alt_pad = ref_pad[:wid // 2] + str(alt) + ref_pad[wid // 2 + len_ref:]

            # one-hot encode the sequences (size=(wid, 4))
            x_ref = one_hot_encode(ref_pad)
            x_alt = one_hot_encode(alt_pad)

            # reverse-complement encoded sequences if the strand is negative
            # (see documentation on `one_hot_encode` to understand why this
            #  works)
            if strand == '-':
                x_ref = x_ref[::-1, ::-1]
                x_alt = x_alt[::-1, ::-1]
            preprocessed_record = PreprocessedAllele(
                record.ref, alt, gene, strand, d_exon_boundary, x_ref, x_alt
            )
            preprocessed_records.append(preprocessed_record)
    return (
        (preprocessed_records, None) if preprocessed_records else
        ([], f'No valid alternative alleles for record: {record}')
    )


def postprocess(dist_var: int, mask: bool, ref: str, alt: str, gene: str,
                strand: str, d_exon_boundary: int,
                y_ref: np.ndarray, y_alt: np.ndarray) -> str:
    """
    Postprocess predictions and pack them into a VCF INFO record.
    This function is heavily based on `get_delta_scores` from the original
    SpliceAI implementation. This is basically a minor refactoring of the
    post-processing part of that function with some comments added to make it
    more readable.
    :param dist_var: maximum distance between the variant and gained/lost splice
    site
    :param mask: mask scores representing annotated acceptor/donor gain and
    unannotated acceptor/donor loss
    :param ref: reference allele
    :param alt: alternative allele
    :param gene: gene name
    :param strand: strand label ('+' or '-')
    :param d_exon_boundary: distance to the closest annotated exon boundary
    :param y_ref: predictions for the reference sequence, shape (npos, 3), where
    npos depends on dist_var
    :param y_alt: predictions fpr the alternative sequence, shape (npos, 3), where
    npos depends on dist_var and the length of alternative allele;
    :return: prediction formatted as a VCF INFO record formatted as
    'ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL', where
    DS_* are delta scores for acceptor gain (AG), acceptor loss (AL), donor
    gain (DG) and donor loss (DL); DP_* are corresponding positions; the scores
    are rounded to 2 digits.
    """
    cov = 2 * dist_var + 1
    len_ref = len(ref)
    len_alt = len(alt)
    len_del = max(len_ref - len_alt, 0)  # deletion length
    # add a dimension to predictions
    y_ref = y_ref[None, :, :]
    y_alt = y_alt[None, :, :]
    # reverse predicted sequence if strand == '-'; this action mirrors the
    # calculation of reverse-complement from `preprocess`
    if strand == '-':
        y_ref = y_ref[:, ::-1]
        y_alt = y_alt[:, ::-1]
    # fill deletions with zeros
    if len_ref > 1 and len_alt == 1:
        y_alt = np.concatenate([
            y_alt[:, :cov // 2 + len_alt],  # predictions before the deletion
            np.zeros((1, len_del, 3)),  # filler
            y_alt[:, cov // 2 + len_alt:]],  # predictions after the deletion
            axis=1)
    # fill insertions with max triplets calculated over the variant-containing
    # slice of the output
    elif len_ref == 1 and len_alt > 1:
        y_alt = np.concatenate([
            y_alt[:, :cov // 2],  # before the variant
            # max calculation and subsequent broadcasting into an array with
            # correct dimensions
            np.max(y_alt[:, cov // 2:cov // 2 + len_alt], axis=1)[:, None, :],
            y_alt[:, cov // 2 + len_alt:]],  # after the variant
            axis=1)
    # concatenate on the 0-th axis -> array with size=(2, cov, 3)
    y = np.concatenate([y_ref, y_alt])
    # the location of the max diff of the 1th position of per-character outputs
    # between the predictions on the reference and alternate sequences
    # (acceptor gain)
    idx_pa = (y[1, :, 1] - y[0, :, 1]).argmax()
    # ... between the alternate and reference sequences (acceptor loss)
    idx_na = (y[0, :, 1] - y[1, :, 1]).argmax()
    # ... the 2nd position of ... (donor gain)
    idx_pd = (y[1, :, 2] - y[0, :, 2]).argmax()
    # ... the 2nd position of ... between the alternate and reference sequences
    # (donor loss)
    idx_nd = (y[0, :, 2] - y[1, :, 2]).argmax()

    mask = int(mask)
    mask_pa = np.logical_and((idx_pa - cov // 2 == d_exon_boundary), mask)
    mask_na = np.logical_and((idx_na - cov // 2 != d_exon_boundary), mask)
    mask_pd = np.logical_and((idx_pd - cov // 2 == d_exon_boundary), mask)
    mask_nd = np.logical_and((idx_nd - cov // 2 != d_exon_boundary), mask)

    return "{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
        alt,
        gene,
        (y[1, idx_pa, 1] - y[0, idx_pa, 1]) * (1 - mask_pa),  # acceptor gain
        (y[0, idx_na, 1] - y[1, idx_na, 1]) * (1 - mask_na),  # acceptor loss
        (y[1, idx_pd, 2] - y[0, idx_pd, 2]) * (1 - mask_pd),  # donor gain
        (y[0, idx_nd, 2] - y[1, idx_nd, 2]) * (1 - mask_nd),  # donor loss
        idx_pa - cov // 2,
        idx_na - cov // 2,
        idx_pd - cov // 2,
        idx_nd - cov // 2
    )


def annotate(nthreads: int, reference: Reference, models: t.List[Model],
             batch_size: int, dist_var: int, mask: bool,
             variants: t.List[VariantRecord]) \
        -> t.List[t.Tuple[t.List[str], t.Optional[str]]]:
    """
    Calculate SpliceAI annotations for list of variants. Annotations are VCF
    info records formatted as
    'ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL', where
    DS_* are delta scores for acceptor gain (AG), acceptor loss (AL), donor
    gain (DG) and donor loss (DL); DP_* are corresponding positions; the scores
    are rounded to 2 digits.
    :param nthreads: the number of CPU threads to use for preprocessing; if
    the reference assembly is stored on a fast-access drive (e.g. an SSD or a
    RAM-disk) using several CPU threads significantly cuts down preprocessing
    time (there is a sizeable improvement up to 4-6 threads with a reference
    stored on an NVME SSD); if the reference is stored on a conventional
    magnetic hard-drive, using multiple threads actually harms performance
    :param reference: a reference assembly and annotation
    :param models: SpliceAI models
    :param batch_size: inference batch size for SpliceAI models; adapt this
    for your GPU(s)
    :param dist_var: maximum distance between the variant and gained/lost splice
    site
    :param mask: mask scores representing annotated acceptor/donor gain and
    unannotated acceptor/donor loss
    :param variants: a list of variants to annotate
    :return: for each variant we return a list of SpliceAI annotations and an
    optional log message
    """
    # preprocess generates a list of PreprocessedAllele objects and an
    # optional logging message for every variant
    if nthreads > 1:
        with ThreadPoolExecutor(nthreads) as workers:
            preprocessed = list(workers.map(partial(preprocess, reference, dist_var), variants))
    else:
        preprocessed = [preprocess(reference, dist_var, var) for var in variants]
    # we need to flatten this list while keeping track of original positions to
    # reconstruct the nested structure later on
    flattened = list(chain.from_iterable(
        [(i, rec) for rec in recs] for i, (recs, _) in enumerate(preprocessed)
    ))
    # alternative sequence have variable length; since Keras models require an
    # array as their input, we will have to sort and group preprocessed alleles
    # by the length of their allele sequences to create valid batches for the
    # models
    length_sorted = sorted(flattened, key=lambda x: x[1].x_alt.shape[0])
    length_groups = groupby(length_sorted, lambda x: x[1].x_alt.shape[0])
    # at this point we can drop indices, becase the ordering will be consistent
    # with `alt_length_sorted`
    length_batches = [list(map(op.itemgetter(1), grp)) for _, grp in length_groups]
    # extract x_ref and x_alt from each batch
    x_ref_batches = [
        np.asarray([rec.x_ref for rec in batch]) for batch in length_batches
    ]
    x_alt_batches = [
        np.asarray([rec.x_alt for rec in batch]) for batch in length_batches
    ]

    def predict_batch(batch: np.ndarray) -> np.ndarray:
        # make prediction with each model
        predictions = [model.predict(batch, batch_size) for model in models]
        # calculate average prediction across models
        return np.mean(predictions, axis=0)

    y_ref_batches = [predict_batch(batch) for batch in x_ref_batches]
    y_alt_batches = [predict_batch(batch) for batch in x_alt_batches]

    # flatten predictions and perform post-processing
    y_ref: t.List[np.ndarray] = list(chain.from_iterable(y_ref_batches))
    y_alt: t.List[np.ndarray] = list(chain.from_iterable(y_alt_batches))

    annotations = [
        (i, postprocess(dist_var, mask, pre.ref, pre.alt, pre.gene, pre.strand,
                        pre.d_exon_boundary, ref, alt))
        for (i, pre), ref, alt in zip(length_sorted, y_ref, y_alt)
    ]
    # group annotations by indices and create a lookup table
    annotations_sorted = sorted(annotations, key=op.itemgetter(0))
    annotations_lookup = {
        i: list(map(op.itemgetter(1), grp))
        for i, grp in groupby(annotations_sorted, key=op.itemgetter(0))
    }
    messages = map(op.itemgetter(1), preprocessed)
    # messages are sorted the same way as input variants, so we can simply
    # iterate enumerate(messages) to get corresponding annotations
    return [(annotations_lookup.get(i, []), message)
            for i, message in enumerate(messages)]


if __name__ == '__main__':
    raise RuntimeError
