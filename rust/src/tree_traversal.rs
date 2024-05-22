use crate::mut_slices::MutSlices;
use crate::selector::Selector;
use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Zip};
use num_traits::AsPrimitive;
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyAnyMethods;
use pyo3::{pyfunction, Bound, FromPyObject, PyResult, Python};
use rayon::iter::{ParallelBridge, ParallelIterator};

type DeltaSumHitCount<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

#[enum_dispatch]
trait DataTrait<'py> {
    fn calc_paths_sum(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        node_offsets: Bound<'py, PyArray1<usize>>,
        weights: Option<Bound<'py, PyArray1<f64>>>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>>;

    fn calc_paths_sum_transpose(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        node_offsets: Bound<'py, PyArray1<usize>>,
        leaf_offsets: Bound<'py, PyArray1<usize>>,
        weights: Option<Bound<'py, PyArray1<f64>>>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>>;

    fn calc_feature_delta_sum(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        node_offsets: Bound<'py, PyArray1<usize>>,
        num_threads: usize,
    ) -> PyResult<DeltaSumHitCount<'py>>;
}

impl<'py, T> DataTrait<'py> for Bound<'py, PyArray2<T>>
where
    T: Element + Copy + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    fn calc_paths_sum(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        node_offsets: Bound<'py, PyArray1<usize>>,
        weights: Option<Bound<'py, PyArray1<f64>>>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let selectors = selectors.readonly();
        let selectors_view = selectors.as_array();
        check_selectors(selectors_view)?;

        let node_offsets = node_offsets.readonly();
        let node_offsets_view = node_offsets.as_array();
        check_node_offsets(node_offsets_view, selectors.len()?)?;

        let data = self.readonly();
        let data_view = data.as_array();
        check_data(data_view)?;

        let weights = weights.map(|weights| weights.readonly());
        let weights_view = weights.as_ref().map(|weights| weights.as_array());

        Ok({
            let paths = PyArray1::zeros_bound(py, data_view.nrows(), false);
            // SAFETY: this call invalidates other views, but it is the only view we need
            let paths_view_mut = unsafe { paths.as_array_mut() };

            // Here we need to dispatch `data` and run the template function
            calc_paths_sum_impl(
                selectors_view,
                node_offsets_view,
                data_view,
                weights_view,
                num_threads,
                paths_view_mut,
            );
            paths
        })
    }

    fn calc_paths_sum_transpose(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        node_offsets: Bound<'py, PyArray1<usize>>,
        leaf_offsets: Bound<'py, PyArray1<usize>>,
        weights: Option<Bound<'py, PyArray1<f64>>>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let selectors = selectors.readonly();
        let selectors_view = selectors.as_array();
        check_selectors(selectors_view)?;

        let node_offsets = node_offsets.readonly();
        let node_offsets_view = node_offsets.as_array();
        check_node_offsets(node_offsets_view, selectors_view.len())?;

        let leaf_offsets = leaf_offsets.readonly();
        let leaf_offsets_view = leaf_offsets.as_array();
        check_leaf_offsets(leaf_offsets_view, node_offsets_view.len())?;

        let data = self.readonly();
        let data_view = data.as_array();
        check_data(data_view)?;

        let weights = weights.map(|weights| weights.readonly());
        let weights_view = weights.as_ref().map(|weights| weights.as_array());

        Ok({
            let values = PyArray1::zeros_bound(
                py,
                *leaf_offsets_view
                    .last()
                    .expect("leaf_offsets array must not be empty"),
                false,
            );

            // SAFETY: this call invalidates other views, but it is the only view we need
            let values_view = unsafe { values.as_array_mut() };

            // Here we need to dispatch `data` and run the template function
            calc_paths_sum_transpose_impl(
                selectors_view,
                node_offsets_view,
                leaf_offsets_view,
                data_view,
                weights_view,
                num_threads,
                values_view,
            );
            values
        })
    }

    fn calc_feature_delta_sum(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        node_offsets: Bound<'py, PyArray1<usize>>,
        num_threads: usize,
    ) -> PyResult<DeltaSumHitCount<'py>> {
        let selectors = selectors.readonly();
        let selectors_view = selectors.as_array();
        check_selectors(selectors_view)?;

        let node_offsets = node_offsets.readonly();
        let node_offsets_view = node_offsets.as_array();
        check_node_offsets(node_offsets_view, selectors.len()?)?;

        let data = self.readonly();
        let data_view = data.as_array();
        check_data(data_view)?;

        Ok({
            let delta_sum =
                PyArray2::zeros_bound(py, (data_view.nrows(), data_view.ncols()), false);
            let hit_count =
                PyArray2::zeros_bound(py, (data_view.nrows(), data_view.ncols()), false);

            // SAFETY: this call invalidates other views, but it is the only view we need
            let delta_sum_view = unsafe { delta_sum.as_array_mut() };
            // SAFETY: this call invalidates other views, but it is the only view we need
            let hit_count_view = unsafe { hit_count.as_array_mut() };

            calc_feature_delta_sum_impl(
                selectors_view,
                node_offsets_view,
                data_view,
                num_threads,
                delta_sum_view,
                hit_count_view,
            );

            (delta_sum, hit_count)
        })
    }
}

#[enum_dispatch(DataTrait)]
#[derive(FromPyObject)]
pub(crate) enum Data<'py> {
    F64(Bound<'py, PyArray2<f64>>),
    F32(Bound<'py, PyArray2<f32>>),
}

// It looks like the performance is not affected by returning a copy of Selector, not reference.
#[inline]
fn find_leaf<T>(tree: &[Selector], sample: &[T]) -> Selector
where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let mut i = 0;
    loop {
        let selector = *unsafe { tree.get_unchecked(i) };
        if selector.is_leaf() {
            break selector;
        }

        // TODO: do opposite type casting: what if we trained on huge f64 and predict on f32?
        let threshold: T = selector.value.as_();
        i = if *unsafe { sample.get_unchecked(selector.feature as usize) } <= threshold {
            selector.left as usize
        } else {
            selector.right as usize
        };
    }
}

#[inline]
fn check_selectors(selectors: ArrayView1<Selector>) -> PyResult<()> {
    if !selectors.is_standard_layout() {
        return Err(PyValueError::new_err(
            "selectors must be contiguous and in memory order",
        ));
    }
    Ok(())
}

#[inline]
fn check_node_offsets(node_offsets: ArrayView1<usize>, selectors_length: usize) -> PyResult<()> {
    if let Some(node_offsets) = node_offsets.as_slice() {
        for (x, y) in node_offsets.iter().copied().tuple_windows() {
            if x > y {
                return Err(PyValueError::new_err(
                    "node_offsets must be sorted in ascending order",
                ));
            }
        }
        if node_offsets[node_offsets.len() - 1] > selectors_length {
            return Err(PyValueError::new_err(
                "node_offsets are out of range of the selectors",
            ));
        }
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "node_offsets must be contiguous and in memory order",
        ))
    }
}

#[inline]
fn check_leaf_offsets(leaf_offsets: ArrayView1<usize>, node_offset_len: usize) -> PyResult<()> {
    if leaf_offsets.len() == 0 {
        return Err(PyValueError::new_err("leaf_offsets must not be empty"));
    }
    if leaf_offsets.len() != node_offset_len {
        return Err(PyValueError::new_err(
            "leaf_offsets must have the same length as node_offsets",
        ));
    }
    if let Some(leaf_offsets) = leaf_offsets.as_slice() {
        for (x, y) in leaf_offsets.iter().copied().tuple_windows() {
            if x > y {
                return Err(PyValueError::new_err(
                    "leaf_offsets must be sorted in ascending order",
                ));
            }
        }
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "leaf_offsets must be contiguous and in memory order",
        ))
    }
}

#[inline]
fn check_data<T>(data: ArrayView2<T>) -> PyResult<()> {
    if !data.is_standard_layout() {
        return Err(PyValueError::new_err(
            "data must be contiguous and in memory order",
        ));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (selectors, node_offsets, data, weights = None, num_threads = 0))]
pub(crate) fn calc_paths_sum<'py>(
    py: Python<'py>,
    selectors: Bound<'py, PyArray1<Selector>>,
    node_offsets: Bound<'py, PyArray1<usize>>,
    // TODO: support f32 data
    data: Data<'py>,
    weights: Option<Bound<'py, PyArray1<f64>>>,
    num_threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    data.calc_paths_sum(py, selectors, node_offsets, weights, num_threads)
}

fn calc_paths_sum_impl<T>(
    selectors: ArrayView1<Selector>,
    node_offsets: ArrayView1<usize>,
    data: ArrayView2<T>,
    weights: Option<ArrayView1<f64>>,
    num_threads: usize,
    paths: ArrayViewMut1<f64>,
) where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let node_offsets = node_offsets.as_slice().unwrap();
    let selectors = selectors.as_slice().unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Cannot build rayon ThreadPool")
        .install(|| {
            Zip::from(paths)
                .and(data.rows())
                .par_for_each(|path, sample| {
                    for (tree_start, tree_end) in node_offsets.iter().copied().tuple_windows() {
                        let tree_selectors =
                            unsafe { selectors.get_unchecked(tree_start..tree_end) };

                        let leaf = find_leaf(tree_selectors, sample.as_slice().unwrap());

                        if let Some(weights) = weights {
                            *path += *unsafe { weights.uget(leaf.left as usize) } * leaf.value;
                        } else {
                            *path += leaf.value;
                        }
                    }
                })
        });
}

#[pyfunction]
#[pyo3(signature = (selectors, node_offsets, leaf_offsets, data, weights = None, num_threads = 0))]
pub(crate) fn calc_paths_sum_transpose<'py>(
    py: Python<'py>,
    selectors: Bound<'py, PyArray1<Selector>>,
    node_offsets: Bound<'py, PyArray1<usize>>,
    leaf_offsets: Bound<'py, PyArray1<usize>>,
    data: Data<'py>,
    weights: Option<Bound<'py, PyArray1<f64>>>,
    num_threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    data.calc_paths_sum_transpose(
        py,
        selectors,
        node_offsets,
        leaf_offsets,
        weights,
        num_threads,
    )
}

fn calc_paths_sum_transpose_impl<T>(
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
        .expect("Cannot get selectors slice from ArrayView");
    let leaf_offsets = leaf_offsets
        .as_slice()
        .expect("Cannot get leaf_offsets slice from ArrayView");

    let values_iter = MutSlices::new(
        values
            .as_slice_mut()
            .expect("values must be contiguous and in memory order"),
        leaf_offsets,
    );

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
                .for_each(|(((tree_start, tree_end), values), &leaf_offset)| {
                    for (x_index, sample) in data.axis_iter(Axis(0)).enumerate() {
                        let tree_selectors =
                            unsafe { selectors.get_unchecked(tree_start..tree_end) };

                        let leaf = find_leaf(tree_selectors, sample.as_slice().unwrap());

                        let value =
                            unsafe { values.get_unchecked_mut(leaf.left as usize - leaf_offset) };
                        if let Some(weights) = weights {
                            *value += weights[x_index] * leaf.value;
                        } else {
                            *value += leaf.value;
                        }
                    }
                })
        });
}

#[pyfunction]
#[pyo3(signature = (selectors, node_offsets, data, num_threads = 0))]
pub(crate) fn calc_feature_delta_sum<'py>(
    py: Python<'py>,
    selectors: Bound<'py, PyArray1<Selector>>,
    node_offsets: Bound<'py, PyArray1<usize>>,
    data: Data<'py>,
    num_threads: usize,
) -> PyResult<DeltaSumHitCount<'py>> {
    data.calc_feature_delta_sum(py, selectors, node_offsets, num_threads)
}

fn calc_feature_delta_sum_impl<T>(
    selectors: ArrayView1<Selector>,
    node_offsets: ArrayView1<usize>,
    data: ArrayView2<T>,
    num_threads: usize,
    mut delta_sum: ArrayViewMut2<f64>,
    mut hit_count: ArrayViewMut2<i64>,
) where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let node_offsets = node_offsets.as_slice().unwrap();
    let selectors = selectors.as_slice().unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Cannot build rayon ThreadPool")
        .install(|| {
            Zip::from(data.rows())
                .and(delta_sum.rows_mut())
                .and(hit_count.rows_mut())
                .par_for_each(|sample, mut delta_sum_row, mut hit_count_row| {
                    for (tree_start, tree_end) in node_offsets.iter().copied().tuple_windows() {
                        let tree_selectors =
                            unsafe { selectors.get_unchecked(tree_start..tree_end) };

                        let mut i = 0;
                        let mut parent_selector: &Selector;
                        loop {
                            parent_selector = unsafe { tree_selectors.get_unchecked(i) };
                            if parent_selector.is_leaf() {
                                break;
                            }

                            // TODO: do opposite type casting: what if we trained on huge f64 and predict on f32?
                            let threshold: T = parent_selector.value.as_();
                            i = if *unsafe { sample.uget(parent_selector.feature as usize) }
                                <= threshold
                            {
                                parent_selector.left as usize
                            } else {
                                parent_selector.right as usize
                            };

                            let child_selector = unsafe { tree_selectors.get_unchecked(i) };
                            // Here we cast to f64 following the original Cython implementation, but
                            // it is a subject to change.
                            *unsafe { delta_sum_row.uget_mut(parent_selector.feature as usize) } +=
                                1.0 + 2.0
                                    * (child_selector.log_n_node_samples
                                        - parent_selector.log_n_node_samples)
                                        as f64;
                            *unsafe { hit_count_row.uget_mut(parent_selector.feature as usize) } +=
                                1;
                        }
                    }
                });
        });
}
