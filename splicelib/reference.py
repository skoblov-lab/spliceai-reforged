import typing as t
import operator as op
from pathlib import Path
from itertools import groupby, count

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from Bio import SeqIO
from ncls import NCLS


class Reference:
    """
    Reference assembly and annotation.
    Assembly attributes:
        assembly - a mapping from chromosome identifiers to chromosome sequences
        long_chrom - True when reference assembly and annotations use long chromosome
            notation format, e.g. "chr1" instead of "1"
    Annotation attributes:
        genes - a numpy array of gene names
        chroms - a numpy array of corresponding chromosome identifiers
        strands - a numpy array of corresponding strand labels ("+" or "-")
        tx_starts - a numpy array of corresponding pre-mRNA transcript starts (1-based)
        tx_ends - a numpy array of corresponding pre-mRNA transcript ends (1-based)
        exon_starts - a list of numpy arrays with corresponding exon start positions (1-based)
        exon_ends - a list of numpy arrays with corresponding exon end positions (1-based)
    Take note, that all annotation attributes have congruent indices. Exon start
    and stop positions are also aligned in pairs.
    """

    def __init__(self,
                 assembly: t.Union[Path, str],
                 annotations: t.Union[Path, str],
                 low_memory: bool = True):
        # read assembly; store assembly in memory if low_memory is false
        if low_memory:
            self.assembly: t.Mapping[str, str] = Fasta(assembly, rebuild=False)
        else:
            self.assembly: t.Mapping[str, str] = {
                seq.id: str(seq.seq)
                for seq in SeqIO.parse(assembly, format='fasta')
            }
        # parse annotations
        try:
            annotations = (
                pd.read_csv(annotations, sep='\t', dtype={'CHROM': str})
                .sort_values(by='CHROM')
            )
            self.genes = annotations['#NAME'].to_numpy()
            self.chroms = annotations['CHROM'].to_numpy()
            self.strands = annotations['STRAND'].to_numpy()
            self.tx_starts = annotations['TX_START'].to_numpy()
            self.tx_ends = annotations['TX_END'].to_numpy()
            self.exon_starts = [
                # TODO the +1 here is specific for GENCODE annotations: GENCODE
                #      EXON_START is actually the last intron position, so +1
                #      moves that to the first exon position. We might want to
                #      modify bundled annotation files to account to this
                np.asarray([int(i) for i in c.split(',') if i]) + 1
                for c in annotations['EXON_START'].to_numpy()
            ]
            self.exon_ends = [
                np.asarray([int(i) for i in c.split(',') if i])
                for c in annotations['EXON_END'].to_numpy()
            ]
        except (KeyError, TypeError, ValueError):
            raise ValueError('incompatible formatting in annotations')
        # make sure reference assembly and annotations have the same chromosome
        # notation style
        asm_has_chrom = any(chrom.startswith('chr') for chrom in self.assembly)
        anno_has_chrom = any(chrom.startswith('chr') for chrom in self.chroms)
        if asm_has_chrom != anno_has_chrom:
            raise ValueError('reference assembly and annotations have different'
                             'chromosome notation format')
        self.long_chrom = asm_has_chrom
        # create annotation indices: one per chromosome;
        # adding 1 to tx_ends to simulate end-inclusive indexing behaviour
        # during intersection lookups in NCLS;
        # count(0) is there to count indices
        records = zip(self.chroms, self.tx_starts, self.tx_ends+1, count(0))

        def extract_intervals(group: t.Iterable[t.Tuple[str, int, int, int]]):
            """
            Given a group (an iterable) of records, each containing a
            chromosome, a gene's start and end coordinates (1-based) and its
            index in feature arrays (i.e. self.chroms, self.strands,
            self.tx_starts, ...), return starts coordinates, end coordinates and
            feature indices as three numpy arrays
            """
            starts, stops, idxs = zip(*map(op.itemgetter(1, 2, 3), group))
            return np.array(starts), np.array(stops), np.array(idxs)

        self._indices = {
            chrom: NCLS(*extract_intervals(group))
            for chrom, group in groupby(records, key=op.itemgetter(0))
        }

    def feature_indices(self, chrom: str, pos: int) -> t.List[int]:
        """
        Return indices of genes covering the position
        :param chrom: a chromosome identifier
        :param pos: 1-based position on the chromosome
        :return:
        """
        intervals = self._indices[chrom]
        return [iv[-1] for iv in intervals.find_overlap(pos, pos)]

    def feature_distances(self, idx: int, pos: int) -> t.Tuple[int, int, int]:
        """
        :param idx: feature index
        :param pos: 1-based position on the feature's chromosome
        :return: distance to transcript start, distance to transcript end,
        distance to the closest exon boundary
        """
        dist_tx_start = self.tx_starts[idx] - pos
        dist_tx_end = self.tx_ends[idx] - pos
        dist_exon_boundary = min(
            np.union1d(self.exon_starts[idx], self.exon_ends[idx]) - pos,
            key=abs
        )
        return dist_tx_start, dist_tx_end, dist_exon_boundary


if __name__ == '__main__':
    raise RuntimeError
