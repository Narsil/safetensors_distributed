use pyo3::Bound as PyBound;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PySlice;
use pyo3::{PyErr, intern};
use safetensors::Dtype;
use safetensors::slice::TensorIndexer;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ops::Bound;

use crate::SafetensorDistributedError;
use crate::loader::PlanSliced;
use crate::loader::dist_loader;

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
    slices: HashMap<String, PlanSliced>,
}

pub(crate) fn slice_to_indexer(
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

impl Default for PyPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyPlan {
    #[new]
    pub fn new() -> Self {
        // PyPlan { inner: Plan::new() }
        PyPlan {
            slices: HashMap::new(),
        }
    }

    pub fn add_slice(&mut self, plan_slice: PlanSliced) -> PyResult<()> {
        match self.slices.entry(plan_slice.name().to_string()) {
            Entry::Vacant(entry) => entry.insert(plan_slice),
            Entry::Occupied(_) => {
                return Err(SafetensorDistributedError::new_err(format!(
                    "slice for {} already exists",
                    plan_slice.name()
                )));
            }
        };
        Ok(())
    }

    pub fn execute(&self, loader: &mut dist_loader) -> PyResult<PyObject> {
        let mut plan = loader.inner.plan();
        for (name, sliced) in self.slices.iter() {
            plan.get_slice(name, sliced.indexers.clone()).unwrap();
        }
        let result = loader
            .runtime
            .block_on(async move { plan.execute().await })
            .map_err(|err| {
                SafetensorDistributedError::new_err(format!("Error during execute {err}"))
            })?;

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (name, tensor) in result.iter() {
                let shape = &tensor.shape;
                let dtype = match tensor.dtype {
                    Dtype::F32 => "float32",
                    Dtype::F64 => "float64",
                    Dtype::I64 => "int64",
                    Dtype::I32 => "int32",
                    Dtype::I16 => "int16",
                    Dtype::I8 => "int8",
                    Dtype::U8 => "uint8",
                    Dtype::BF16 => "bfloat16",
                    Dtype::F16 => "float16",
                    _ => return Err(SafetensorDistributedError::new_err("Unsupported dtype")),
                };
                let numpy = py.import("numpy")?;
                let array = numpy
                    .getattr("frombuffer")?
                    .call1((tensor.data.clone(), numpy.getattr(dtype)?))?;
                let array = array.call_method1("reshape", (shape,))?;
                dict.set_item(name, array)?;
            }
            Ok(dict.into())
        })
    }
}
