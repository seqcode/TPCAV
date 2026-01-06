"""
Lightweight, reusable TCAV utilities built from the repository scripts.

The package keeps existing scripts untouched while offering programmatic
access to concept construction and PCA/attribution workflows.
"""

import logging

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

from .cavs import CavTrainer
from .concepts import ConceptBuilder
from .helper import (
    bed_to_chrom_tracks_iter,
    bed_to_fasta_iter,
    dataframe_to_chrom_tracks_iter,
    dataframe_to_fasta_iter,
    dinuc_shuffle_sequences,
    fasta_to_one_hot_sequences,
    random_regions_dataframe,
)
from .logging_utils import set_verbose
from .tpcav_model import TPCAV

__all__ = [
    "ConceptBuilder",
    "CavTrainer",
    "TPCAV",
    "bed_to_fasta_iter",
    "dataframe_to_fasta_iter",
    "bed_to_chrom_tracks_iter",
    "dataframe_to_chrom_tracks_iter",
    "fasta_to_one_hot_sequences",
    "random_regions_dataframe",
    "dinuc_shuffle_sequences",
    "set_verbose",
]
