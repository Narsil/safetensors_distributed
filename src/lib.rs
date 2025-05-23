use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reqwest::Client;
use reqwest::Url;
use safetensors::tensor::Metadata;
use safetensors::{SafeTensorError, SafeTensors};
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

const MAX_HEADER_SIZE: usize = 100_000_000;

#[pyclass]
struct SafeTensorsLoader {
    url: Url,
    client: Arc<Client>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl SafeTensorsLoader {
    #[new]
    fn new(url: String) -> PyResult<Self> {
        let url = Url::parse(&url)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid URL: {} ({})", e, url)))?;
        let client = Arc::new(Client::new());
        let runtime = Arc::new(Runtime::new().unwrap());
        Ok(SafeTensorsLoader {
            url,
            client,
            runtime,
        })
    }

    fn __enter__(&self) -> PyResult<Self> {
        Ok(self.clone())
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

    fn get_metadata(&self) -> PyResult<()> {
        let client = Arc::clone(&self.client);
        let url = self.url.clone();
        let runtime = Arc::clone(&self.runtime);

        let buffer = runtime
            .block_on(async move {
                let header_size = 10 * 1024 * 1024; // 10 MB
                let response = client
                    .get(url)
                    .header("Range", format!("bytes=0-{}", header_size - 1))
                    .send()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok::<Vec<u8>, PyErr>(bytes.to_vec())
            })
            .map_err(|e| PyErr::from(e))?;

        let metadata = get_metadata(buffer)
            .map_err(|err| SafetensorDistributedError::new_err("Error loading header : {err:?}"))?;
        Ok(())
    }
}

fn get_metadata(buffer: Vec<u8>) -> Result<Metadata, SafeTensorError> {
    let buffer_len = buffer.len();
    if buffer_len < 8 {
        return Err(SafeTensorError::HeaderTooSmall);
    }
    let arr: [u8; 8] = [
        buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
    ];
    let n: usize = u64::from_le_bytes(arr)
        .try_into()
        .map_err(|_| SafeTensorError::HeaderTooLarge)?;
    if n > MAX_HEADER_SIZE {
        return Err(SafeTensorError::HeaderTooLarge);
    }

    let stop = n
        .checked_add(8)
        .ok_or(SafeTensorError::InvalidHeaderLength)?;
    if stop > buffer_len {
        return Err(SafeTensorError::InvalidHeaderLength);
    }
    let string =
        core::str::from_utf8(&buffer[8..stop]).map_err(|_| SafeTensorError::InvalidHeader)?;
    // Assert the string starts with {
    // NOTE: Add when we move to 0.4.0
    // if !string.starts_with('{') {
    //     return Err(SafeTensorError::InvalidHeaderStart);
    // }
    let metadata: Metadata =
        serde_json::from_str(string).map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;
    Ok(metadata)
}
impl Clone for SafeTensorsLoader {
    fn clone(&self) -> Self {
        SafeTensorsLoader {
            url: self.url.clone(),
            client: Arc::clone(&self.client),
            runtime: Arc::clone(&self.runtime),
        }
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
        let client = Arc::clone(&loader.client);
        let url = loader.url.clone();
        let slices = self.slices.clone();
        let runtime = Arc::clone(&loader.runtime);

        let result = runtime
            .block_on(async move {
                let response = client
                    .get(url)
                    .send()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let mut result = HashMap::new();
                for (tensor_name, _) in slices {
                    result.insert(tensor_name, bytes.to_vec());
                }
                Ok::<HashMap<String, Vec<u8>>, PyErr>(result)
            })
            .map_err(|e| PyErr::from(e))?;

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
    PyException,
    "Custom Python Exception for Safetensor Distributed errors."
);

#[pymodule]
fn safetensors_distributed(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SafeTensorsLoader>()?;
    module.add_class::<Plan>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_dummy() {
        assert_eq!(2 + 2, 4);
    }
}
