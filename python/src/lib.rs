use pyo3::Bound as PyBound;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

pyo3::create_exception!(
    safetensors_distributed,
    SafetensorDistributedError,
    PyException
);

#[pymodule]
fn safetensors_distributed(module: &PyBound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "SafetensorDistributedError",
        module.py().get_type::<SafetensorDistributedError>(),
    )?;
    Ok(())
}
