#!/usr/bin/env python

from traits.api \
    import HasTraits, Array, Str, List, Dict

# -----------------------------------------------------------#
# Models


class Dataset(HasTraits):
    x = Array
    y = Array
    title = Str
    info = Dict


class CorrectedDatasets(HasTraits):
    datasets = List(Dataset)
    title = Str
    info = Dict


class Measurement(HasTraits):
    corrected_datasets = List(CorrectedDatasets)
    title = Str


class Experiment(HasTraits):
    measurements = List(Measurement)
    title = Str
    info = Dict
