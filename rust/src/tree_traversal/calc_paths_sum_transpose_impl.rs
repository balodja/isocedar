use crate::mut_slices::MutSlices;
use crate::selector::Selector;
use crate::tree_traversal::find_leaf::find_leaf;
use itertools::{Either, Itertools};
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use num_traits::AsPrimitive;
use rayon::prelude::*;
use std::iter::repeat;

pub(super) fn calc_paths_sum_transpose_impl<T>(
    selectors: ArrayView1<Selector>,
    node_offsets: ArrayView1<usize>,
    leaf_offsets: ArrayView1<usize>,
    data: ArrayView2<T>,
    weights: Option<ArrayView1<f64>>,
    num_threads: usize,
    mut values: ArrayViewMut1<f64>,
) where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let selectors = selectors
        .as_slice()
        .expect("selectors must be contiguous and in memory order");
    let leaf_offsets = leaf_offsets
        .as_slice()
        .expect("leaf_offsets must be contiguous and in memory order");
    let weights = weights.as_ref().map(|array_view| {
        array_view
            .as_slice()
            .expect("weights must be contiguous and in memory order")
    });
    let values = values
        .as_slice_mut()
        .expect("values must be contiguous and in memory order");

    if num_threads == 1 {
        single_thread(selectors, node_offsets, data, weights, values)
    } else {
        multithread(
            selectors,
            node_offsets,
            leaf_offsets,
            data,
            weights,
            num_threads,
            values,
        )
    }
}

fn single_thread<T>(
    selectors: &[Selector],
    node_offsets: ArrayView1<usize>,
    data: ArrayView2<T>,
    weights: Option<&[f64]>,
    values: &mut [f64],
) where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    for (sample, weight) in data.axis_iter(Axis(0)).zip(weights_iterator(weights)) {
        for tree_range in node_offsets.iter().copied().tuple_windows() {
            update_values(selectors, sample, weight, tree_range, values, 0)
        }
    }
}

fn multithread<T>(
    selectors: &[Selector],
    node_offsets: ArrayView1<usize>,
    leaf_offsets: &[usize],
    data: ArrayView2<T>,
    weights: Option<&[f64]>,
    num_threads: usize,
    values: &mut [f64],
) where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let values_iter = MutSlices::new(values, leaf_offsets);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Cannot build rayon ThreadPool")
        .install(|| {
            node_offsets
                .iter()
                .copied()
                .tuple_windows()
                .zip(values_iter)
                .zip(leaf_offsets)
                .par_bridge()
                .for_each(|((tree_range, values), &leaf_offset)| {
                    for (sample, weight) in data.axis_iter(Axis(0)).zip(weights_iterator(weights)) {
                        update_values(selectors, sample, weight, tree_range, values, leaf_offset)
                    }
                });
        });
}

fn weights_iterator(weights: Option<&[f64]>) -> impl Iterator<Item = f64> + '_ {
    match weights {
        Some(weights) => Either::Left(weights.iter().copied()),
        None => Either::Right(repeat(1.0)),
    }
}

fn update_values<T>(
    selectors: &[Selector],
    sample: ArrayView1<T>,
    weight: f64,
    tree_range: (usize, usize),
    values: &mut [f64],
    leaf_offset: usize,
) where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let (tree_start, tree_end) = tree_range;
    let tree_selectors = unsafe { selectors.get_unchecked(tree_start..tree_end) };
    let leaf = find_leaf(tree_selectors, sample.as_slice().unwrap());
    *unsafe { values.get_unchecked_mut(leaf.left as usize - leaf_offset) } += weight * leaf.value;
}
