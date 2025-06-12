use pyo3::Bound as PyBound;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::path::Path;

pyo3::create_exception!(
    safetensors_distributed,
    SafetensorDistributedError,
    PyException
);

/// Fuse a distributed safetensors checkpoint into a single shared model.
///
/// Args:
///     input_dir: Path to the input directory containing topology.json and rank*.safetensors files
///     output_dir: Path to the output directory where model.safetensors will be written
///
/// Raises:
///     SafetensorDistributedError: If there is an error during fusion
#[pyfunction]
fn fuse_checkpoint(input_dir: &str, output_dir: &str) -> PyResult<()> {
    let _input_path = Path::new(input_dir);
    let _output_path = Path::new(output_dir);

    // TODO: Call the Rust implementation
    // This will be implemented later in the Rust code
    Ok(())
}

#[pymodule]
fn safetensors_distributed(module: &PyBound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "SafetensorDistributedError",
        module.py().get_type::<SafetensorDistributedError>(),
    )?;
    module.add_function(wrap_pyfunction!(fuse_checkpoint, module)?)?;
    Ok(())
}
