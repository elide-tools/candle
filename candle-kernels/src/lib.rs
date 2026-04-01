mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

/// `true` when the crate was built without nvcc. In this mode, [`Module::ptx()`]
/// returns raw CUDA source (`.cu`) rather than compiled PTX. The consumer
/// must compile it at runtime via nvrtc before loading it onto a device.
pub const RUNTIME_COMPILE: bool = cfg!(candle_kernels_runtime_compile);

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    /// Pre-compiled PTX when nvcc was available at build time,
    /// or raw CUDA source (`.cu`) when `RUNTIME_COMPILE` is true.
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns either pre-compiled PTX or raw CUDA source, depending on
    /// whether nvcc was available at build time. Check [`RUNTIME_COMPILE`]
    /// to determine which variant this is.
    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);

#[cfg(feature = "moe")]
pub mod ffi;
