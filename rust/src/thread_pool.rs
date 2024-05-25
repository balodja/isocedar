use pyo3::exceptions::PyRuntimeError;
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

import_exception!(pickle, PicklingError);
import_exception!(pickle, UnpicklingError);

#[derive(Clone, Serialize, Deserialize)]
#[serde(try_from = "ThreadPoolSerde", into = "ThreadPoolSerde")]
#[pyclass]
struct ThreadPool {
    inner: Arc<rayon::ThreadPool>,
}

impl ThreadPool {
    fn new(n_jobs: usize) -> PyResult<Self> {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs)
            .build()
            .map_err(|err| {
                PyRuntimeError::new_err(format!(
                    r#"Error happened on the Rust side when creating ThreadPool: "{err}""#
                ))
            })?;
        Ok(Self {
            inner: Arc::new(thread_pool),
        })
    }
}

#[pymethods]
impl ThreadPool {
    #[new]
    fn __new__(n_jobs: usize) -> PyResult<Self> {
        Self::new(n_jobs)
    }

    /// Used by pickle.load / pickle.loads
    fn __setstate__(&mut self, state: Bound<PyBytes>) -> PyResult<()> {
        *self = serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
            .map_err(|err| {
                UnpicklingError::new_err(format!(
                    r#"Error happened on the Rust side when deserializing ThreadPool: "{err}""#
                ))
            })?;
        Ok(())
    }

    /// Used by pickle.dump / pickle.dumps
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let vec_bytes =
            serde_pickle::to_vec(&self, serde_pickle::SerOptions::new()).map_err(|err| {
                PicklingError::new_err(format!(
                    r#"Error happened on the Rust side when serializing ThreadPool: "{err}""#
                ))
            })?;
        Ok(PyBytes::new_bound(py, &vec_bytes))
    }

    /// Used by copy.copy, makes a shallow copy
    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Used by copy.deepcopy, makes a new instance
    fn __deepcopy__(&self, _memo: Bound<PyAny>) -> PyResult<Self> {
        let thread_pool_serde: ThreadPoolSerde = self.clone().into();
        thread_pool_serde.try_into()
    }
}

#[derive(Serialize, Deserialize)]
struct ThreadPoolSerde {
    n_jobs: usize,
}

impl From<ThreadPool> for ThreadPoolSerde {
    fn from(thread_pool: ThreadPool) -> Self {
        Self {
            n_jobs: thread_pool.inner.current_num_threads(),
        }
    }
}

impl TryFrom<ThreadPoolSerde> for ThreadPool {
    type Error = PyErr;

    fn try_from(thread_pool_serde: ThreadPoolSerde) -> Result<Self, Self::Error> {
        Self::new(thread_pool_serde.n_jobs)
    }
}
