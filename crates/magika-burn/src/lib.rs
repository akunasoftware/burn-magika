mod config;
mod detection;
mod preprocess;
mod runtime;

pub use config::{LabelEntry, ModelConfig};
pub use detection::{Detection, RankedAlternative};
pub use preprocess::preprocess_bytes;
pub use runtime::{
    InferenceRuntime, MagikaClassifier, MagikaClassifierBuilder, MagikaInferenceError,
};
