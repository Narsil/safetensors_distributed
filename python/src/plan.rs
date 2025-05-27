use pyo3::Bound as PyBound;
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3::{PyErr, intern};
use safetensors::slice::TensorIndexer;
use std::ops::Bound;

use crate::SafetensorDistributedError;
use crate::loader::PySliced;

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

// use std::fmt;

// struct Disp(Vec<TensorIndexer>);
//
// /// Should be more readable that the standard
// /// `Debug`
// impl fmt::Display for Disp {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "[")?;
//         for (i, item) in self.0.iter().enumerate() {
//             if i != self.0.len() - 1 {
//                 write!(f, "{item}, ")?;
//             } else {
//                 write!(f, "{item}")?;
//             }
//         }
//         write!(f, "]")
//     }
// }

#[pyclass]
pub struct PyPlan {
    // inner: Plan,
    // slices:
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
    pub fn new() -> Self {
        // PyPlan { inner: Plan::new() }
        PyPlan {}
    }

    pub fn add_slice(&mut self, py_slices: PySliced) -> PyResult<()> {
        todo!();
        // let slices: Slice = py_slices.extract()?;
        // let is_list = py_slices.is_instance_of::<PyList>();

        // let slices: Vec<SliceIndex> = match slices {
        //     Slice::Slice(slice) => vec![slice],
        //     Slice::Slices(slices) => {
        //         if slices.is_empty() && is_list {
        //             vec![SliceIndex::Slice(PySlice::new(py_slices.py(), 0, 0, 0))]
        //         } else if is_list {
        //             return Err(SafetensorDistributedError::new_err(
        //                 "Non empty lists are not implemented",
        //             ));
        //         } else {
        //             slices
        //         }
        //     }
        // };
        // loader.runtime.block_on(async {
        //     let (metadata, _offset) = loader.inner.metadata().await.map_err(|err| {
        //         SafetensorDistributedError::new_err(format!("Failed to get metadata: {err}"))
        //     })?;
        //     let info = metadata.info(&tensor_name).ok_or_else(|| {
        //         SafetensorDistributedError::new_err(format!(
        //             "Missing tensor {}",
        //             tensor_name.to_string()
        //         ))
        //     })?;
        //     let shape = info.shape.clone();
        //     let slices: Vec<TensorIndexer> = slices
        //         .into_iter()
        //         .zip(shape)
        //         .enumerate()
        //         .map(slice_to_indexer)
        //         .collect::<Result<_, _>>()?;
        //     self.inner
        //         .get_slice(tensor_name, slices)
        //         .await
        //         .map_err(|err| {
        //             SafetensorDistributedError::new_err(format!("Failed to get metadata: {err}"))
        //         })
        // })?;

        // Ok(())
    }

    // pub fn execute(&self, loader: &mut dist_loader) -> PyResult<PyObject> {
    //     let result = loader.runtime.block_on(async {
    //         self.inner.execute(&mut loader.inner).await.map_err(|err| {
    //             SafetensorDistributedError::new_err(format!("Failed to execute plan: {err}"))
    //         })
    //     })?;

    //     Python::with_gil(|py| {
    //         let dict = PyDict::new(py);
    //         for (name, tensor) in result {
    //             let shape = tensor.shape;
    //             let dtype = match tensor.dtype {
    //                 Dtype::F32 => "float32",
    //                 Dtype::F64 => "float64",
    //                 Dtype::I64 => "int64",
    //                 Dtype::I32 => "int32",
    //                 Dtype::I16 => "int16",
    //                 Dtype::I8 => "int8",
    //                 Dtype::U8 => "uint8",
    //                 Dtype::BF16 => "bfloat16",
    //                 Dtype::F16 => "float16",
    //                 _ => return Err(SafetensorDistributedError::new_err("Unsupported dtype")),
    //             };
    //             let numpy = py.import("numpy")?;
    //             let array = numpy
    //                 .getattr("frombuffer")?
    //                 .call1((tensor.data, numpy.getattr(dtype)?))?;
    //             let array = array.call_method1("reshape", (shape,))?;
    //             dict.set_item(name, array)?;
    //         }
    //         Ok(dict.into())
    //     })
    // }
}
