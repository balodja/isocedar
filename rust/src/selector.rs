use numpy::{Element, PyArrayDescr};
use pyo3::prelude::{PyAnyMethods, PyDictMethods};
use pyo3::sync::GILOnceCell;
use pyo3::types::PyDict;
use pyo3::{py_run, Bound, Py, PyResult, Python};

static SELECTOR_DTYPE_CELL: GILOnceCell<Py<PyArrayDescr>> = GILOnceCell::new();

/// Selector is the representation of decision tree nodes: either branches or leafs.
///
/// We use "C"-representation with standard alignment (np.dtype(align=True)), but "packed"
/// (dtype(aligh=False)) would work as well.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct Selector {
    /// Feature index to branch on, -1.0 if leaf
    pub(crate) feature: i32,
    /// Index of left subtree, leaf_id if leaf
    pub(crate) left: i32,
    /// Feature value to branch on, resulting decision score if leaf
    pub(crate) value: f64,
    /// Index of right subtree, -1 if leaf
    pub(crate) right: i32,
    /// Natural logarithm of the number of samples in the node
    pub(crate) log_n_node_samples: f32,
}

impl Selector {
    pub(crate) fn dtype(py: Python) -> PyResult<Bound<PyArrayDescr>> {
        let unbind_dtype =
            SELECTOR_DTYPE_CELL.get_or_try_init(py, || -> PyResult<_> {
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
            .downcast::<PyArrayDescr>()?.clone()
            .unbind())
            })?;
        Ok(unbind_dtype.bind(py).clone())
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
