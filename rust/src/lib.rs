use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use num_traits::AsPrimitive;
use numpy::PyArrayMethods;
use numpy::{Element, PyArray, PyArrayDescr};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::py_run;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::iter;
use std::sync::{Arc, Mutex};

/// Selector is the representation of decision tree nodes: either branches or leafs.
///
/// We use "C"-representation with standard alignment (np.dtype(align=True)), but "packed"
/// (dtype(aligh=False)) would work as well.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct Selector {
    /// Feature index to branch on, -1.0 if leaf
    feature: i32,
    /// Index of left subtree, leaf_id if leaf
    left: i32,
    /// Feature value to branch on, resulting decision score if leaf
    value: f64,
    /// Index of right subtree, -1 if leaf
    right: i32,
    /// Natural logarithm of the number of samples in the node
    log_n_node_samples: f32,
}

impl Selector {
    pub(crate) fn dtype(py: Python) -> PyResult<Bound<PyArrayDescr>> {
        let locals = PyDict::new_bound(py);
        py_run!(
            py,
            *locals,
            r#"
            dtype = __import__('numpy').dtype(
                [
                    ('feature', 'i4'),
                    ('left', 'i4'),
                    ('value', 'f8'),
                    ('right', 'i4'),
                    ('log_n_node_samples', 'f4')
                ],
                align=True,
            )
            "#
        );
        Ok(locals
            .get_item("dtype")
            .expect("Error in built-in Python code for dtype initialization")
            .expect("Error in built-in Python code for dtype initialization: dtype cannot be None")
            .downcast::<PyArrayDescr>()?
            .clone())
    }

    #[inline(always)]
    pub(crate) fn is_leaf(&self) -> bool {
        self.feature == -1
    }
}

/// Implementation of [numpy::Element] for [Selector]
///
/// Safety: we guarantee that [Selector] has the same layout as it would have in numpy with
/// [Selector::dtype]
unsafe impl Element for Selector {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python) -> Bound<PyArrayDescr> {
        Self::dtype(py).unwrap()
    }
}

#[enum_dispatch]
trait DataTrait<'py> {
    fn calc_paths_sum(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        indices: Bound<'py, PyArray1<i64>>,
        weights: Option<Bound<'py, PyArray1<f64>>>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>>;

    fn calc_paths_sum_transpose(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        indices: Bound<'py, PyArray1<i64>>,
        leaf_count: usize,
        weights: Option<Bound<'py, PyArray1<f64>>>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>>;

    fn calc_feature_delta_sum(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        indices: Bound<'py, PyArray1<i64>>,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)>;
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
        indices: Bound<'py, PyArray1<i64>>,
        weights: Option<Bound<'py, PyArray1<f64>>>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let selectors = selectors.readonly();
        let selectors_view = selectors.as_array();
        check_selectors(selectors_view)?;

        let indices = indices.readonly();
        let indices_view = indices.as_array();
        check_indices(indices_view, selectors.len()?)?;

        let data = self.readonly();
        let data_view = data.as_array();
        check_data(data_view)?;

        let weights = weights.map(|weights| weights.readonly());
        let weights_view = weights.as_ref().map(|weights| weights.as_array());

        // Here we need to dispatch `data` and run the template function
        let values = calc_paths_sum_impl(
            selectors_view,
            indices_view,
            data_view,
            weights_view,
            num_threads,
        );
        Ok(PyArray::from_owned_array_bound(py, values))
    }

    fn calc_paths_sum_transpose(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        indices: Bound<'py, PyArray1<i64>>,
        leaf_count: usize,
        weights: Option<Bound<'py, PyArray1<f64>>>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let selectors = selectors.readonly();
        let selectors_view = selectors.as_array();
        check_selectors(selectors_view)?;

        let indices = indices.readonly();
        let indices_view = indices.as_array();
        check_indices(indices_view, selectors.len()?)?;

        let data = self.readonly();
        let data_view = data.as_array();
        check_data(data_view)?;

        let weights = weights.map(|weights| weights.readonly());
        let weights_view = weights.as_ref().map(|weights| weights.as_array());

        // Here we need to dispatch `data` and run the template function
        let values = crate::calc_paths_sum_transpose_impl(
            selectors_view,
            indices_view,
            leaf_count,
            data_view,
            weights_view,
            num_threads,
        );
        Ok(PyArray::from_owned_array_bound(py, values))
    }

    fn calc_feature_delta_sum(
        &self,
        py: Python<'py>,
        selectors: Bound<'py, PyArray1<Selector>>,
        indices: Bound<'py, PyArray1<i64>>,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)> {
        let selectors = selectors.readonly();
        let selectors_view = selectors.as_array();
        check_selectors(selectors_view)?;

        let indices = indices.readonly();
        let indices_view = indices.as_array();
        check_indices(indices_view, selectors.len()?)?;

        let data = self.readonly();
        let data_view = data.as_array();
        check_data(data_view)?;

        let (delta_sum, hit_count) =
            calc_feature_delta_sum_impl(selectors_view, indices_view, data_view, num_threads);

        let delta_sum = PyArray::from_owned_array_bound(py, delta_sum);
        let hit_count = PyArray::from_owned_array_bound(py, hit_count);

        Ok((delta_sum, hit_count))
    }
}

#[enum_dispatch(DataTrait)]
#[derive(FromPyObject)]
enum Data<'py> {
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
fn check_indices(indices: ArrayView1<i64>, selectors_length: usize) -> PyResult<()> {
    if let Some(indices) = indices.as_slice() {
        for (x, y) in indices.iter().copied().tuple_windows() {
            if x > y {
                return Err(PyValueError::new_err(
                    "indices must be sorted in ascending order",
                ));
            }
        }
        if indices[indices.len() - 1] as usize > selectors_length {
            return Err(PyValueError::new_err(
                "indices are out of range of the selectors",
            ));
        }
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "indices must be contiguous and in memory order",
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
#[pyo3(signature = (selectors, indices, data, weights = None, num_threads = 0))]
pub(crate) fn calc_paths_sum<'py>(
    py: Python<'py>,
    selectors: Bound<'py, PyArray1<Selector>>,
    indices: Bound<'py, PyArray1<i64>>,
    // TODO: support f32 data
    data: Data<'py>,
    weights: Option<Bound<'py, PyArray1<f64>>>,
    num_threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    data.calc_paths_sum(py, selectors, indices, weights, num_threads)
}

fn calc_paths_sum_impl<T>(
    selectors: ArrayView1<Selector>,
    indices: ArrayView1<i64>,
    data: ArrayView2<T>,
    weights: Option<ArrayView1<f64>>,
    num_threads: usize,
) -> Array1<f64>
where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let mut paths = Array1::zeros(data.nrows());

    let indices = indices.as_slice().unwrap();
    let selectors = selectors.as_slice().unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Cannot build rayon ThreadPool")
        .install(|| {
            Zip::from(paths.view_mut())
                .and(data.rows())
                .par_for_each(|path, sample| {
                    for (tree_start, tree_end) in
                        indices.iter().map(|i| *i as usize).tuple_windows()
                    {
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

    paths
}

#[pyfunction]
#[pyo3(signature = (selectors, indices, data, leaf_count, weights = None, num_threads = 0))]
pub(crate) fn calc_paths_sum_transpose<'py>(
    py: Python<'py>,
    selectors: Bound<'py, PyArray1<Selector>>,
    indices: Bound<'py, PyArray1<i64>>,
    data: Data<'py>,
    leaf_count: usize,
    weights: Option<Bound<'py, PyArray1<f64>>>,
    num_threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    data.calc_paths_sum_transpose(py, selectors, indices, leaf_count, weights, num_threads)
}

fn calc_paths_sum_transpose_impl<T>(
    selectors: ArrayView1<Selector>,
    indices: ArrayView1<i64>,
    leaf_count: usize,
    data: ArrayView2<T>,
    weights: Option<ArrayView1<f64>>,
    num_threads: usize,
) -> Array1<f64>
where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    // We need leaf_offsets instead of leaf_counts here.
    // It would allow to split the array and write safely from multiple threads.
    let values = Arc::new((0..leaf_count).map(|_| Mutex::new(0.0)).collect::<Vec<_>>());

    let selectors = selectors.as_slice().unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Cannot build rayon ThreadPool")
        .install(|| {
            indices
                .iter()
                .map(|i| *i as usize)
                .tuple_windows()
                .zip(iter::repeat_with(|| values.clone()))
                .par_bridge()
                .for_each(|((tree_start, tree_end), values)| {
                    for (x_index, sample) in data.axis_iter(Axis(0)).enumerate() {
                        let tree_selectors =
                            unsafe { selectors.get_unchecked(tree_start..tree_end) };

                        let leaf = find_leaf(tree_selectors, sample.as_slice().unwrap());

                        let mut value = values[leaf.left as usize].lock().unwrap();
                        if let Some(weights) = weights {
                            *value += weights[x_index] * leaf.value;
                        } else {
                            *value += leaf.value;
                        }
                    }
                })
        });

    Arc::try_unwrap(values)
        .unwrap()
        .into_iter()
        .map(|mutex| mutex.into_inner().unwrap())
        .collect()
}

#[pyfunction]
#[pyo3(signature = (selectors, indices, data, num_threads = 0))]
pub(crate) fn calc_feature_delta_sum<'py>(
    py: Python<'py>,
    selectors: Bound<'py, PyArray1<Selector>>,
    indices: Bound<'py, PyArray1<i64>>,
    data: Data<'py>,
    num_threads: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)> {
    data.calc_feature_delta_sum(py, selectors, indices, num_threads)
}

fn calc_feature_delta_sum_impl<T>(
    selectors: ArrayView1<Selector>,
    indices: ArrayView1<i64>,
    data: ArrayView2<T>,
    num_threads: usize,
) -> (Array2<f64>, Array2<i64>)
where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let indices = indices.as_slice().unwrap();
    let selectors = selectors.as_slice().unwrap();

    let mut delta_sum = Array2::zeros((data.nrows(), data.ncols()));
    let mut hit_count = Array2::zeros((data.nrows(), data.ncols()));

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Cannot build rayon ThreadPool")
        .install(|| {
            Zip::from(data.rows())
                .and(delta_sum.rows_mut())
                .and(hit_count.rows_mut())
                .par_for_each(|sample, mut delta_sum_row, mut hit_count_row| {
                    for (tree_start, tree_end) in
                        indices.iter().map(|i| *i as usize).tuple_windows()
                    {
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
                            *unsafe { delta_sum_row.uget_mut(parent_selector.feature as usize) } +=
                                1.0 + 2.0
                                    * (child_selector.log_n_node_samples as f64
                                        - parent_selector.log_n_node_samples as f64);
                            *unsafe { hit_count_row.uget_mut(parent_selector.feature as usize) } +=
                                1;
                        }
                    }
                });
        });

    (delta_sum, hit_count)
}

#[pymodule]
#[pyo3(name = "calc_paths_sum")]
fn rust_module(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("selector_dtype", Selector::dtype(_py)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum, m)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(calc_feature_delta_sum, m)?)?;
    Ok(())
}
