//! File-type detection using Magika and Burn.
//!
//! # Example
//!
//! ```rust,no_run
//! use burn_magika::Session;
//! use burn_wgpu::{Wgpu, WgpuDevice};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let device = WgpuDevice::default();
//!     let mut session = Session::<Wgpu>::new(&device)?;
//!
//!     let detected = session.identify_content_sync(b"fn main() { println!(\"hi\"); }")?;
//!     println!("{} {}", detected.info().label, detected.info().mime_type);
//!
//!     Ok(())
//! }
//! ```

mod config;
mod content {
    pub use crate::vendor::content::*;
}
mod detection;
mod file;
mod model;
mod preprocess;
mod session;
mod vendor;

pub use config::ModelConfig;
pub use detection::{Detection, RankedAlternative};
pub use file::{FileType, InferredType, OverwriteReason, TypeInfo};
pub use model::MagikaInferenceError as Error;
pub use model::{MagikaInferenceError, MagikaModel};
pub use preprocess::preprocess_bytes;
pub use session::Session;
pub use vendor::content::{ContentType, MODEL_MAJOR_VERSION, MODEL_NAME};
