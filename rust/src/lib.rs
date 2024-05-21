mod mut_slices;
mod selector;
mod tree_traversal;

use crate::selector::Selector;
use crate::tree_traversal::{calc_feature_delta_sum, calc_paths_sum, calc_paths_sum_transpose};
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "calc_paths_sum")]
fn rust_module(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("selector_dtype", Selector::dtype(py)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum, m)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(calc_feature_delta_sum, m)?)?;
    Ok(())
}
