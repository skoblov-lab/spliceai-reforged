import os
import typing as t
import logging
from pathlib import Path

import click
import pysam
from tensorflow.keras.models import Model, load_model

from splicelib import trained_models, builtin_annotations
from splicelib.reference import Reference
from splicelib.processing import annotate
from splicelib.utils import iterate_batches

try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# N_VISIBLE_GPUS = len(tf.config.experimental.list_physical_devices('GPU'))
ANNOTATIONS = {
    'grch37': resources.path(builtin_annotations, 'grch37.txt'),
    'grch38': resources.path(builtin_annotations, 'grch38.txt')
}


def load_models() -> t.List[Model]:
    models_ = []
    for i in range(1, 6):
        with resources.path(trained_models, f'spliceai{i}.h5') as path:
            models_.append(load_model(path))
    return models_


@click.command('spliceai', help='Annotate a VCF file with SpliceAI predictions')
@click.option('-i', '--input', required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help='Path to the input VCF file')
@click.option('-o', '--output', required=True,
              type=click.Path(exists=False, writable=True, resolve_path=True),
              help='Path to the output VCF file')
@click.option('-r', '--ref_assembly', required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help='Path to a reference assembly. For best performance the '
                   'assembly should be stored on a fast-access drive (an SSD '
                   'or a RAM-disk). Chromosome identifiers in the assembly '
                   'must follow the same notation format as chromosome values in '
                   'the annotation file. Built-in annotations use short-format '
                   'chromosome notation (that is there are no "chr" prefixes '
                   'in chromosome identifiers).')
@click.option('-a', '--annotations', required=True, type=str,
              help='Reference genome annotation. You can specify a path to '
                   'an annotation file or specify one of built-in annotations: '
                   '"grch37" or "grch38". An annotation file must contain the '
                   'following columns: #NAME (gene name), CHROM (chromosome), '
                   'STRAND, TX_START (transcript start), TX_END (transcript end) '
                   'EXON_START (exon start positions), EXON_END (exon end '
                   'positions). The values in CHROM must follow the same '
                   'chromosome naming format as sequence identifiers in '
                   'the reference assembly. Built-in annotations use '
                   'short-format chromosome notation (that is there are no '
                   '"chr" prefixes in chromosome identifiers). TX_START and '
                   'TX_END use 1-based indexing, end positions are '
                   'inclusive. EXON_START and EXON_END are comma-separated '
                   'lists of corresponding exon start and end locations. We '
                   'use GENCODE annotations, wherein EXON_START values point to '
                   'the last intron position right next to an exon. Your custom '
                   'annotation files must follow the same conventions. '
                   'Annotation files must be tab-separated.')
@click.option('-d', '--distance', type=click.IntRange(0, 5000), default=50,
              help='Maximum distance between the variant and gained/lost '
                   'splice site. An integer in the range [0, 5000]. Defaults '
                   'to 50.')
@click.option('--mask', is_flag=True, type=bool,
              help='Activate the masking of scores representing annotated '
                   'acceptor/donor gain and unannotated acceptor/donor loss')
@click.option('--preprocessing_threads', default=1,
              type=click.IntRange(1, 4, clamp=True),
              help='The number of preprocessing threads to use. If your '
                   'reference assembly is located on a fast-access drive '
                   '(an SSD or a RAM-disk), using up to 4 preprocessing '
                   'threads significantly cuts down preprocessing time. '
                   'Defaults to 1.')
@click.option('--preprocessing_batch', default=10000,
              type=click.IntRange(1, None, clamp=True),
              help='The number of variant records in a VCF file to process at '
                   'a time. Using larger batches tends to cut down '
                   'preprocessing overhead. It is safe to keep the default '
                   'value. Defaults to 10000')
@click.option('--prediction_batch', default=64,
              type=click.IntRange(1, None, clamp=True),
              help='The batch size to use during inference. This option is '
                   'useful if you are using a GPU and want to exploit its full '
                   'potential. It is best to use powers of 2. If you are '
                   'getting out of memory errors, you should reduce the '
                   'batch size. Defaults to 64')
def spliceai(input, output, ref_assembly, annotations, distance, mask,
             preprocessing_threads, preprocessing_batch, prediction_batch):
    # parse reference assembly and annotations
    try:
        with ANNOTATIONS[annotations] as anno:
            reference = Reference(ref_assembly, anno)
    except KeyError:
        anno = Path(annotations).absolute()
        if not anno.exists():
            raise click.BadOptionUsage(
                'annotations', f'annotation file {anno} does not exist'
            )
        reference = Reference(ref_assembly, anno)
    # load models
    models = load_models()
    # open the input, update the header and open the output
    vcf_input = pysam.VariantFile(input)
    header = vcf_input.header
    header.add_line(
        '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
        'annotation. These include delta scores (DS) and delta positions (DP) for '
        'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
        'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">'
    )
    vcf_output = pysam.VariantFile(output, mode='w', header=header)
    # break input into batches
    input_batches = iterate_batches(preprocessing_batch, vcf_input)
    for batch in input_batches:
        # for every input variant `annotate` returns a list of annotation
        # strings and an optional logging message
        scores = annotate(
            preprocessing_threads, reference, models, prediction_batch,
            distance, mask, batch
        )
        for variant, (scores_, message) in zip(batch, scores):
            if message:
                logging.error(message)
            variant.info['SpliceAI'] = scores_
            vcf_output.write(variant)


if __name__ == '__main__':
    spliceai()
