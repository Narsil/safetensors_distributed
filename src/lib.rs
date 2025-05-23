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
use tokio::task::{JoinError, JoinHandle};

const MAX_HEADER_SIZE: usize = 100_000_000;

#[pyclass]
struct SafeTensorsLoader {
    url: Url,
    client: Arc<Client>,
    runtime: Arc<Runtime>,
    metadata_future: Option<JoinHandle<Result<Metadata, SafeTensorError>>>,
    metadata: Option<Metadata>,
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("The future didn't exist")]
    NoFuture,
    #[error(transparent)]
    JoinError(#[from] JoinError),
    #[error(transparent)]
    Safetensor(#[from] SafeTensorError),
}

#[pymethods]
impl SafeTensorsLoader {
    #[new]
    fn new(url: String) -> PyResult<Self> {
        let url = Url::parse(&url)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid URL: {} ({})", e, url)))?;
        let client = Arc::new(Client::new());
        let runtime = Arc::new(Runtime::new().unwrap());

        let fut_client = Arc::clone(&client);
        let fut_url = url.clone();
        let metadata_future = Some(runtime.spawn(async move {
            let header_size = 10 * 1024 * 1024; // 10 MB
            let response = fut_client
                .get(fut_url)
                .header("Range", format!("bytes=0-{}", header_size - 1))
                .send()
                .await
                .map_err(|e| SafeTensorError::InvalidHeader)?;
            let bytes = response
                .bytes()
                .await
                .map_err(|e| SafeTensorError::InvalidHeader)?;
            get_metadata(bytes.to_vec())
        }));

        Ok(SafeTensorsLoader {
            url,
            client,
            runtime,
            metadata_future,
            metadata: None,
        })
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
        let metadata = self
            .get_metadata()
            .map_err(|e| SafetensorDistributedError::new_err(e.to_string()))?;
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
    let metadata: Metadata =
        serde_json::from_str(string).map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;
    Ok(metadata)
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
impl SafeTensorsLoader {
    fn get_metadata(&mut self) -> Result<&Metadata, Error> {
        if self.metadata.is_none() {
            let future = self.metadata_future.take().ok_or_else(|| Error::NoFuture)?;

            let fut_result = self.runtime.block_on(future)?;
            let metadata: Metadata = fut_result?;
            self.metadata = Some(metadata);
        }
        Ok(self.metadata.as_ref().unwrap())
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
    use super::*;
    use std::fs;

    #[test]
    fn test_open_safetensors_file() {
        let buffer = fs::read("test.safetensors").expect("Failed to read test.safetensors");
        let metadata = get_metadata(buffer).expect("Failed to parse metadata");
        println!("Metadata: {:?}", metadata);
    }
}
