mod config;
mod detection;
mod file;
mod model;
mod preprocess;
mod vendor;

pub use config::ModelConfig;
pub use detection::{Detection, RankedAlternative};
pub use model::{MagikaInferenceError, MagikaModel};
pub use preprocess::preprocess_bytes;
pub use vendor::content::{ContentType, MODEL_MAJOR_VERSION, MODEL_NAME};
