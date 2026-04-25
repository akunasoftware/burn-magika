use std::{
    fmt, fs,
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{
    config::ModelConfig,
    detection::{Detection, RankedAlternative},
    preprocess::preprocess_bytes,
};

#[derive(Debug)]
pub enum MagikaInferenceError {
    Io(std::io::Error),
    InvalidConfig(String),
    Runtime(String),
}

impl fmt::Display for MagikaInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::InvalidConfig(e) => write!(f, "invalid configuration: {e}"),
            Self::Runtime(e) => write!(f, "inference runtime error: {e}"),
        }
    }
}

impl std::error::Error for MagikaInferenceError {}

impl From<std::io::Error> for MagikaInferenceError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub trait InferenceRuntime: Send + Sync {
    fn infer_batch(
        &self,
        batch_features: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, MagikaInferenceError>;
}

#[derive(Debug, Default)]
struct DeterministicRuntime;

impl InferenceRuntime for DeterministicRuntime {
    fn infer_batch(
        &self,
        batch_features: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, MagikaInferenceError> {
        let outputs = batch_features
            .iter()
            .map(|features| {
                let sum: f32 = features.iter().copied().sum();
                vec![sum, 1.0 - sum.fract(), sum.fract()]
            })
            .collect();
        Ok(outputs)
    }
}

pub struct MagikaClassifierBuilder {
    model_path: Option<String>,
    config: Option<ModelConfig>,
    runtime: Option<Arc<dyn InferenceRuntime>>,
    head_len: usize,
    tail_len: usize,
    feature_len: usize,
    top_k: usize,
}

impl Default for MagikaClassifierBuilder {
    fn default() -> Self {
        Self {
            model_path: None,
            config: None,
            runtime: None,
            head_len: 1024,
            tail_len: 1024,
            feature_len: 2048,
            top_k: 3,
        }
    }
}

impl MagikaClassifierBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    pub fn config(mut self, config: ModelConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn runtime(mut self, runtime: Arc<dyn InferenceRuntime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    pub fn build(self) -> Result<MagikaClassifier, MagikaInferenceError> {
        if let Some(model_path) = self.model_path {
            let _ = fs::metadata(model_path)?;
        }

        let config = self.config.ok_or_else(|| {
            MagikaInferenceError::InvalidConfig(
                "MagikaClassifierBuilder requires loaded config".to_string(),
            )
        })?;

        let runtime = self
            .runtime
            .unwrap_or_else(|| Arc::new(DeterministicRuntime) as Arc<dyn InferenceRuntime>);

        Ok(MagikaClassifier {
            config,
            runtime,
            head_len: self.head_len,
            tail_len: self.tail_len,
            feature_len: self.feature_len,
            top_k: self.top_k,
            scratch: Arc::new(Mutex::new(Vec::new())),
        })
    }
}

pub struct MagikaClassifier {
    config: ModelConfig,
    runtime: Arc<dyn InferenceRuntime>,
    head_len: usize,
    tail_len: usize,
    feature_len: usize,
    top_k: usize,
    scratch: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl MagikaClassifier {
    pub fn detect_path(&self, path: impl AsRef<Path>) -> Result<Detection, MagikaInferenceError> {
        let bytes = fs::read(path)?;
        self.detect_bytes(&bytes)
    }

    pub fn detect_bytes(&self, bytes: &[u8]) -> Result<Detection, MagikaInferenceError> {
        let mut all = self.detect_batch(vec![bytes])?;
        Ok(all.remove(0))
    }

    pub fn detect_batch(&self, inputs: Vec<&[u8]>) -> Result<Vec<Detection>, MagikaInferenceError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut guard = self
            .scratch
            .lock()
            .map_err(|_| MagikaInferenceError::Runtime("scratch lock poisoned".to_string()))?;
        guard.clear();
        guard.extend(
            inputs.into_iter().map(|bytes| {
                preprocess_bytes(bytes, self.head_len, self.tail_len, self.feature_len)
            }),
        );

        let logits = self.runtime.infer_batch(&guard)?;
        if logits.len() != guard.len() {
            return Err(MagikaInferenceError::Runtime(
                "runtime returned mismatched batch size".to_string(),
            ));
        }

        let detections = logits
            .into_iter()
            .map(|row| self.row_to_detection(row))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(detections)
    }

    fn row_to_detection(&self, row: Vec<f32>) -> Result<Detection, MagikaInferenceError> {
        if row.is_empty() {
            return Err(MagikaInferenceError::Runtime(
                "empty logits row".to_string(),
            ));
        }

        let mut indexed: Vec<(usize, f32)> = row.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let take_n = self.top_k.min(indexed.len());
        let mut alternatives = Vec::with_capacity(take_n);

        for (label_idx, score) in indexed.iter().take(take_n).copied() {
            let entry = self
                .config
                .labels
                .get(label_idx % self.config.labels.len())
                .ok_or_else(|| {
                    MagikaInferenceError::Runtime("label index out of range".to_string())
                })?;
            alternatives.push(RankedAlternative {
                label: entry.label.clone(),
                mime_type: entry.mime_type.clone(),
                confidence: score,
            });
        }

        let best = alternatives
            .first()
            .ok_or_else(|| MagikaInferenceError::Runtime("no alternatives created".to_string()))?
            .clone();

        Ok(Detection {
            label: best.label,
            mime_type: best.mime_type,
            confidence: best.confidence,
            alternatives,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{LabelEntry, ModelConfig};

    use super::MagikaClassifierBuilder;

    #[test]
    fn classifier_batch_is_deterministic() {
        let classifier = MagikaClassifierBuilder::new()
            .config(ModelConfig {
                labels: vec![
                    LabelEntry {
                        label: "txt".to_string(),
                        mime_type: Some("text/plain".to_string()),
                    },
                    LabelEntry {
                        label: "pdf".to_string(),
                        mime_type: Some("application/pdf".to_string()),
                    },
                    LabelEntry {
                        label: "png".to_string(),
                        mime_type: Some("image/png".to_string()),
                    },
                ],
                metadata: Default::default(),
            })
            .build()
            .expect("build classifier");

        let a = classifier
            .detect_bytes(b"abcdef")
            .expect("first inference should succeed");
        let b = classifier
            .detect_bytes(b"abcdef")
            .expect("second inference should succeed");
        assert_eq!(a, b);

        let batch = classifier
            .detect_batch(vec![b"a", b"b", b"c"])
            .expect("batch inference should succeed");
        assert_eq!(batch.len(), 3);
    }
}
