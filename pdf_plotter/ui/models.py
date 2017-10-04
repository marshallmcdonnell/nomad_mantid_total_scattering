#!/usr/bin/env python

from traits.api \
    import HasTraits, Array, Str, List

# -----------------------------------------------------------#
# Models


class Dataset(HasTraits):
    x = Array
    y = Array
    title = Str


class Measurement(HasTraits):
    datasets = List(Dataset)
    title = Str


class Experiment(HasTraits):
    measurements = List(Measurement)
    title = Str
