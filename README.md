# spliceai-reforged
Another implementation of SpliceAI. This one is designed to better utilise GPU-parallelism than the [original implementation](https://github.com/Illumina/SpliceAI). it only makes sense to use this implementation if you are planning to use a GPU to predict a lot of variants. If you are going to use this implementation in you research, don't forget to reference the original paper.

```
Usage: spliceai.py [OPTIONS]

  Annotate a VCF file with SpliceAI predictions

Options:
  -i, --input FILE                Path to the input VCF file  [required]
  -o, --output PATH               Path to the output VCF file  [required]
  -r, --ref_assembly FILE         Path to a reference assembly. For best
                                  performance the assembly should be stored on
                                  a fast-access drive (an SSD or a RAM-disk).
                                  Chromosome identifiers in the assembly must
                                  follow the same notation format as
                                  chromosome values in the annotation file.
                                  Built-in annotations use short-format
                                  chromosome notations (that is there are no
                                  "chr" prefixes in chromosome identifiers).
                                  [required]

  -a, --annotations TEXT          Reference genome annotations. You can
                                  specify a path to an annotation file or
                                  specify one of built-in annotations:
                                  "grch37" and "grch38". An annotation file
                                  must contain the following columns: #NAME
                                  (gene name), CHROM (chromosome), STRAND,
                                  TX_START (transcript start), TX_END
                                  (transcript end) EXON_START (exon start
                                  positions), EXON_END (exon end positions).
                                  The values in CHROM must follow the same
                                  chromosome naming format as sequence
                                  identifiers in the reference assembly.
                                  Built-in annotations use short-format
                                  chromosome notations (that is there are no
                                  "chr" prefixes in chromosome identifiers).
                                  TX_START and TX_END are use 1-based
                                  indexing, end positions are inclusive.
                                  EXON_START and EXON_END are comma-separated
                                  lists of corresponding exon start and end
                                  location. We use GENCODE annotations,
                                  wherein EXON_START values point to the last
                                  intron position right next to an exon. Your
                                  custom annotation files must follow the same
                                  conventions. Annotation files must be tab-
                                  separated.  [required]

  -d, --distance INTEGER RANGE    Maximum distance between the variant and
                                  gained/lost splice site. An integer in the
                                  range [0, 5000]. Defaults to 50.

  --mask                          Activate the masking of scores representing
                                  annotated acceptor/donor gain and
                                  unannotated acceptor/donor loss

  --preprocessing_threads INTEGER RANGE
                                  The number of preprocessing threads to use.
                                  If your reference sequence is located on a
                                  fast-access drive (an SSD or a RAM-disk),
                                  using up to 4 preprocessing threads
                                  significantly cuts down preprocessing time.
                                  Defaults to 1.

  --preprocessing_batch INTEGER RANGE
                                  The number of variant records in a VCF file
                                  to process at a time. Using larger batches
                                  tends to cut down preprocessing overhead. It
                                  is safe to keep the default value. Defaults
                                  to 10000

  --prediction_batch INTEGER RANGE
                                  The batch size to use during inference. This
                                  option is useful if you are using a GPU and
                                  want to use its full potential. It is best
                                  to use powers of 2. If you are getting out
                                  of memory errors, you should reduce the
                                  batch size. Defaults to 64

  --help                          Show this message and exit.
```
