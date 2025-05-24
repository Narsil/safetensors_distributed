use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reqwest::Url;
use serde_json;
use std::collections::HashMap;

mod loader;
use loader::{Loader, Error};

#[pyclass]
struct SafeTensorsLoader {
    inner: Loader,
}

#[pymethods]
impl SafeTensorsLoader {
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

    fn create_plan(&self) -> PyResult<Plan> {
        Ok(Plan::new())
    }

    pub fn metadata(&mut self) -> PyResult<PyObject> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let metadata = rt.block_on(self.inner.metadata()).map_err(|err| {
            SafetensorDistributedError::new_err(format!("Failed to get metadata: {err}"))
        })?;
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

#[pyclass]
struct Plan {
    slices: HashMap<String, Vec<(usize, usize)>>,
}

#[pymethods]
impl Plan {
    #[new]
    fn new() -> Self {
        Plan {
            slices: HashMap::new(),
        }
    }

    fn add_slice(&mut self, tensor_name: String, start: usize, end: usize) -> PyResult<()> {
        self.slices
            .entry(tensor_name)
            .or_insert_with(Vec::new)
            .push((start, end));
        Ok(())
    }

    fn execute(&self, loader: &SafeTensorsLoader) -> PyResult<PyObject> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(loader.inner.execute(self)).map_err(|err| {
            SafetensorDistributedError::new_err(format!("Could not execute plan: {err}"))
        })?;
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (key, value) in result {
                dict.set_item(key, value)?;
            }
            Ok(dict.into())
        })
    }
}

pyo3::create_exception!(
    safetensors_distributed,
    SafetensorDistributedError,
    PyException
);

#[pymodule]
fn safetensors_distributed(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SafeTensorsLoader>()?;
    module.add_class::<Plan>()?;
    module.add(
        "SafetensorDistributedError",
        module.getattr("SafetensorDistributedError")?,
    )?;
    Ok(())
}
