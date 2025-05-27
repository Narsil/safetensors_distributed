use pyo3::Bound as PyBound;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use reqwest::Url;
use safetensors::slice::TensorIndexer;
use safetensors::tensor::{Metadata, TensorInfo};
use serde_json;
use tokio::runtime::Runtime;

use safetensors_distributed::loader::Loader;

use crate::SafetensorDistributedError;
use crate::plan::PyPlan;

#[pyclass]
#[allow(non_camel_case_types)]
pub struct dist_loader {
    inner: Loader,
    runtime: Runtime,
}

#[pymethods]
impl dist_loader {
    #[new]
    fn new(url: String) -> PyResult<Self> {
        let url = Url::parse(&url).map_err(|e| {
            SafetensorDistributedError::new_err(format!("Invalid URL: {} ({})", e, url))
        })?;
        let runtime = Runtime::new().map_err(|e| {
            SafetensorDistributedError::new_err(format!("Failed to create runtime: {}", e))
        })?;
        let handle = runtime.handle().clone();
        let inner = Loader::new_py(url, handle).map_err(|err| {
            SafetensorDistributedError::new_err(format!("Failed to get create loader: {err}"))
        })?;
        Ok(Self { inner, runtime })
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

    fn plan(&self) -> PyResult<PyPlan> {
        Ok(PyPlan::new())
    }

    fn get_slice(&mut self, tensor_name: String) -> PyResult<PySlice> {
        let (metadata, _) = self.raw_metadata()?;
        let info = metadata.info(&tensor_name).ok_or_else(|| {
            SafetensorDistributedError::new_err(format!("{tensor_name:?} does not exist"))
        })?;
        Ok(PySlice::new(info.clone()))
    }

    pub fn metadata(&mut self) -> PyResult<PyObject> {
        let (metadata, _offset) = self.raw_metadata()?;
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            if let Some(user_metadata) = metadata.metadata() {
                let json: serde_json::Value = serde_json::to_value(user_metadata)
                    .map_err(|e| SafetensorDistributedError::new_err(e.to_string()))?;
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

impl dist_loader {
    fn raw_metadata(&mut self) -> PyResult<&(Metadata, usize)> {
        let metadata = self
            .runtime
            .block_on(async { self.inner.metadata().await })
            .map_err(|err| {
                SafetensorDistributedError::new_err(format!("Failed to get metadata: {err}"))
            })?;
        Ok(metadata)
    }
}

#[pyclass]
#[allow(non_camel_case_types)]
pub struct PySlice {
    info: TensorInfo,
}

#[pymethods]
impl PySlice {
    fn __getitem__(&self, pyslices: PyBound<'_, PyAny>) -> PySliced {
        todo!()
    }
}
impl PySlice {
    fn new(info: TensorInfo) -> Self {
        Self { info }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySliced {
    info: TensorInfo,
    indexers: Vec<TensorIndexer>,
}

impl PySliced {
    fn new(info: TensorInfo, indexers: Vec<TensorIndexer>) -> Self {
        Self { info, indexers }
    }
}
