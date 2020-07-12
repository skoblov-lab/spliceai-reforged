# spliceai-reforged
Another implementation of SpliceAI. This one is designed to better utilise GPU-parallelism than the [original implementation](https://github.com/Illumina/SpliceAI). Trained models were taken directly from the original implementation, so you are going to get the same predictions with this implementation. It only makes sense to use this implementation if you are planning to use a GPU to predict a lot of variants. If you are going to use this implementation in your research, don't forget to reference the original paper [Jaganathan et al, Cell 2019 in press](https://doi.org/10.1016/j.cell.2018.12.015).

## Installation

Although we've added an automatic `pip` installer that grabs all the required dependencies, the best way to install `spliceai-reforged` is to create a new conda environment and grab dependencies from conda repositories rather than pip's repository. The following instructions assume you are planning to use a GPU. If that's not the case, replace `tensorflow-gpu` with `tensorflow` (though in that case you might as well use the original implementation).

```
$ conda create -y -n spliceai python=3.7 numpy pandas tensorflow-gpu=2.2 click
$ conda activate spliceai
$ conda install -y -c bioconda pysam pyfaidx ncls
$ pip install --no-cache-dir git+https://github.com/skoblov-lab/spliceai-reforged.git
```
You can run use `spliceai.py`

## Usage

Example run with bundled data.
```
spliceai.py -r grch37.fna -a grch37 -i testdata/input.vcf -o testdata/output.vcf 
```

## Documentation

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
                                  chromosome notation (that is there are no
                                  "chr" prefixes in chromosome identifiers).
                                  [required]

  -a, --annotations TEXT          Reference genome annotation. You can specify
                                  a path to an annotation file or specify one
                                  of built-in annotations: "grch37" or
                                  "grch38". An annotation file must contain
                                  the following columns: #NAME (gene name),
                                  CHROM (chromosome), STRAND, TX_START
                                  (transcript start), TX_END (transcript end)
                                  EXON_START (exon start positions), EXON_END
                                  (exon end positions). The values in CHROM
                                  must follow the same chromosome naming
                                  format as sequence identifiers in the
                                  reference assembly. Built-in annotations use
                                  short-format chromosome notation (that is
                                  there are no "chr" prefixes in chromosome
                                  identifiers). TX_START and TX_END use
                                  1-based indexing, end positions are
                                  inclusive. EXON_START and EXON_END are
                                  comma-separated lists of corresponding exon
                                  start and end locations. We use GENCODE
                                  annotations, wherein EXON_START values point
                                  to the last intron position right next to an
                                  exon. Your custom annotation files must
                                  follow the same conventions. Annotation
                                  files must be tab-separated.  [required]

  -d, --distance INTEGER RANGE    Maximum distance between the variant and
                                  gained/lost splice site. An integer in the
                                  range [0, 5000]. Defaults to 50.

  --mask                          Activate the masking of scores representing
                                  annotated acceptor/donor gain and
                                  unannotated acceptor/donor loss

  --preprocessing_threads INTEGER RANGE
                                  The number of preprocessing threads to use.
                                  If your reference assembly is located on a
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
                                  want to exploit its full potential. It is
                                  best to use powers of 2. If you are getting
                                  out of memory errors, you should reduce the
                                  batch size. Defaults to 64

  --help                          Show this message and exit.
```
