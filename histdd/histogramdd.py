# numpy
import numpy as np

# dask
import dask.array as da
from dask.delayed import Delayed, unpack_collections
import dask.base
import dask.core
from dask.highlevelgraph import HighLevelGraph
## dirty import of private function
from dask.array.routines import _linspace_from_delayed

# we might want to use range as a kwarg so alias the built-in.
_RANGE = range


def _block_histogramdd(sample, bins, range=None, weights=None):
    """Blocked numpy.histogramdd calculation.

    Slurps the result into another axis via [np.newaxis]. This new
    axis is used to stack chunked results and add them together later.

    """
    return np.histogramdd(x, bins, range=range, weights=weights)[0][np.newaxis]


def dask_histogramdd(sample, bins, range, weights=None, density=None):
    """Histogram data in multiple dimensions

    Dask version of :func:`np.histogramdd`.

    Current prototype requires bins to be a tuple of ints (total
    number of bins in each dimension) and range to to be a tuple of
    (xmin, xmax) entries (the range of each dimension).

    A lot of type/shape checking logic will be required for a
    production implementation. Weights are currently unsupported as
    well.

    Parameters
    ----------
    sample : dask Array
        Multidimensional data to histogram.
    bins : sequence of ints
        Number of bins in each dimension to histogram.
    range : sequence of pairs
        Axes limits in each dimension (xmin, xmax)
    weights : dask Array, optional
        Optional weights associated with the data. (TODO)
    density : bool
        Convert bins to values of a PDF (TODO)

    See Also
    --------
    histogram

    """

    # N == total number of samples
    # D == total number of dimensions
    N, D = sample.shape

    # generate token and name for task
    token = dask.base.tokenize(sample, bins, range, weights, density)
    name = "histogramdd-sum-" + token

    ######################################################################
    ## Convert the bin and range definitions for each axis into a
    ## delayed array of bin edges (for each dimension). In this
    ## prototype the bins are _not_ delayed, naming is taking into
    ## consider future compat. with delayed bin edge arrays.
    possibly_delayed_bins = []
    for b, r in zip(bins, range):
        possibly_delayed_bins.append(_linspace_from_delayed(r[0], r[1], b + 1))
    ## total number of bin edge arrays must be the total number of dimensions
    assert len(possibly_delayed_bins) == D
    (bins_ref, range_ref), deps = unpack_collections([possibly_delayed_bins, range])

    ######################################################################
    # New graph where the callable is _block_histogramdd and the
    # function arguments are chunks of our sample array and then the
    # bin information. The graph is stacked for each chunk, each
    # building block is a D-dimensional histogram result.
    dsk = {
        (name, i, 0): (_block_histogramdd, k, bins_ref, range_ref)
        for i, k in enumerate(dask.core.flatten(sample.__dask_keys__()))
    }
    # da.histogram does this to get the dtype (not obvious why)
    dtype = np.histogramdd([])[0].dtype
    # da.histogram does this to track possible dependencies from
    # potentially delayed binning information.
    deps = (sample,) + deps
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)

    ######################################################################
    ## this logic yields chunks where the shape is:
    ## (nchunks, nbin_D1, nbin_D2, nbin_D3, ...)
    ## such that we add (collapse) the chunks along axis=0 to get the
    ## final result.
    ##
    ## steps:
    ## 1. get the total number of chunks in the sample
    ## 2. make a tuple of tuples; each inner tuple is the total number
    ##    of bins in a dimension
    ## 3. make "stacked chunks" tuple which combines information from
    ##    steps 1 and 2
    ## 4. make dask Array and collapse the 0th axis (which has size
    ##    nchunks_in_sample)
    nchunks_in_sample = len(list(dask.core.flatten(sample.__dask_keys__())))
    all_nbins = tuple((b.size - 1,) for b in possibly_delayed_bins)
    stacked_chunks = ((1,) * nchunks_in_sample, *all_nbins)
    mapped = da.Array(graph, name, stacked_chunks, dtype=dtype)
    # finally sum over sample chunk providing the final result array.
    n = mapped.sum(axis=0)

    # TODO: density operation logic (just some division).

    return n, [b for b in possibly_delayed_bins]
