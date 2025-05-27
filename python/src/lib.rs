use pyo3::Bound as PyBound;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

mod loader;
pub mod plan;
use loader::dist_loader;
use plan::PyPlan;

pyo3::create_exception!(
    safetensors_distributed,
    SafetensorDistributedError,
    PyException
);

#[pymodule]
fn safetensors_distributed(module: &PyBound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<dist_loader>()?;
    module.add_class::<PyPlan>()?;
    module.add(
        "SafetensorDistributedError",
        module.py().get_type::<SafetensorDistributedError>(),
    )?;
    Ok(())
}
