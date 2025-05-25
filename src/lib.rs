use pyo3::Bound as PyBound;
use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice};
use pyo3::{PyErr, intern};
use reqwest::Url;
use safetensors::slice::TensorIndexer;
use safetensors::tensor::Metadata;
use serde_json;
use std::ops::Bound;

mod loader;
mod plan;
use loader::Loader;
use plan::Plan;

#[derive(FromPyObject)]
pub(crate) enum SliceIndex<'a> {
    Slice(PyBound<'a, PySlice>),
    Index(i32),
}

#[derive(FromPyObject)]
pub(crate) enum Slice<'a> {
    Slice(SliceIndex<'a>),
    Slices(Vec<SliceIndex<'a>>),
}

use std::fmt;

struct Disp(Vec<TensorIndexer>);

/// Should be more readable that the standard
/// `Debug`
impl fmt::Display for Disp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.0.iter().enumerate() {
            if i != self.0.len() - 1 {
                write!(f, "{item}, ")?;
            } else {
                write!(f, "{item}")?;
            }
        }
        write!(f, "]")
    }
}

#[pyclass]
struct PyLoader {
    inner: Loader,
}

#[pyclass]
struct PyPlan {
    inner: Plan,
}

fn slice_to_indexer(
    (dim_idx, (slice_index, dim)): (usize, (SliceIndex, usize)),
) -> Result<TensorIndexer, PyErr> {
    match slice_index {
        SliceIndex::Slice(slice) => {
            let py_start = slice.getattr(intern!(slice.py(), "start"))?;
            let start: Option<usize> = py_start.extract()?;
            let start = if let Some(start) = start {
                Bound::Included(start)
            } else {
                Bound::Unbounded
            };

            let py_stop = slice.getattr(intern!(slice.py(), "stop"))?;
            let stop: Option<usize> = py_stop.extract()?;
            let stop = if let Some(stop) = stop {
                Bound::Excluded(stop)
            } else {
                Bound::Unbounded
            };
            Ok(TensorIndexer::Narrow(start, stop))
        }
        SliceIndex::Index(idx) => {
            if idx < 0 {
                let idx = dim.checked_add_signed(idx as isize).ok_or(
                    SafetensorDistributedError::new_err(format!(
                        "Invalid index {idx} for dimension {dim_idx} of size {dim}"
                    )),
                )?;
                Ok(TensorIndexer::Select(idx))
            } else {
                Ok(TensorIndexer::Select(idx as usize))
            }
        }
    }
}

#[pymethods]
impl PyPlan {
    #[new]
    fn new() -> Self {
        PyPlan { inner: Plan::new() }
    }

    pub fn get_slice(
        &mut self,
        loader: &mut PyLoader,
        tensor_name: &str,
        py_slices: PyBound<'_, PyAny>,
    ) -> PyResult<()> {
        let slices: Slice = py_slices.extract()?;
        let is_list = py_slices.is_instance_of::<PyList>();

        let slices: Vec<SliceIndex> = match slices {
            Slice::Slice(slice) => vec![slice],
            Slice::Slices(slices) => {
                if slices.is_empty() && is_list {
                    vec![SliceIndex::Slice(PySlice::new(py_slices.py(), 0, 0, 0))]
                } else if is_list {
                    return Err(SafetensorDistributedError::new_err(
                        "Non empty lists are not implemented",
                    ));
                } else {
                    slices
                }
            }
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let info = loader
                .inner
                .metadata()
                .await
                .map_err(|err| {
                    SafetensorDistributedError::new_err(format!("Failed to get metadata: {err}"))
                })?
                .info(&tensor_name)
                .ok_or_else(|| {
                    SafetensorDistributedError::new_err(format!(
                        "Missing tensor {}",
                        tensor_name.to_string()
                    ))
                })?;
            let shape = info.shape.clone();
            let slices: Vec<TensorIndexer> = slices
                .into_iter()
                .zip(shape)
                .enumerate()
                .map(slice_to_indexer)
                .collect::<Result<_, _>>()?;
            self.inner
                .get_slice(tensor_name, slices)
                .await
                .map_err(|err| {
                    SafetensorDistributedError::new_err(format!("Failed to get metadata: {err}"))
                })
        })?;

        Ok(())
    }
}

#[pymethods]
impl PyLoader {
    #[new]
    fn new(url: String) -> PyResult<Self> {
        let url = Url::parse(&url)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid URL: {} ({})", e, url)))?;
        let inner = Loader::new(url).map_err(|err| {
            SafetensorDistributedError::new_err(format!("Failed to get create loader: {err}"))
        })?;
        Ok(Self { inner })
    }

    /// Start the context manager
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> PyResult<()> {
        Ok(())
    }

    fn create_plan(&self) -> PyResult<PyPlan> {
        Ok(PyPlan::new())
    }

    pub fn metadata(&mut self) -> PyResult<PyObject> {
        let metadata = self.raw_metadata()?;
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            if let Some(user_metadata) = metadata.metadata() {
                let json: serde_json::Value = serde_json::to_value(user_metadata)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                if let Some(obj) = json.as_object() {
                    for (k, v) in obj.iter() {
                        dict.set_item(k, v.to_string())?;
                    }
                }
            }
            Ok(dict.into())
        })
    }
}

impl PyLoader {
    fn raw_metadata(&mut self) -> PyResult<&Metadata> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let metadata = rt.block_on(self.inner.metadata()).map_err(|err| {
            SafetensorDistributedError::new_err(format!("Failed to get metadata: {err}"))
        })?;
        Ok(metadata)
    }
}

pyo3::create_exception!(
    safetensors_distributed,
    SafetensorDistributedError,
    PyException
);

#[pymodule]
fn safetensors_distributed(module: &PyBound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLoader>()?;
    module.add_class::<PyPlan>()?;
    module.add(
        "SafetensorDistributedError",
        module.getattr("SafetensorDistributedError")?,
    )?;
    Ok(())
}
