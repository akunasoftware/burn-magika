use std::{collections::HashMap, fs, path::Path};

use crate::runtime::MagikaInferenceError;

#[derive(Debug, Clone)]
pub struct LabelEntry {
    pub label: String,
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub labels: Vec<LabelEntry>,
    pub metadata: HashMap<String, String>,
}

impl ModelConfig {
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self, MagikaInferenceError> {
        let raw = fs::read_to_string(path)?;
        Self::from_json_str(&raw)
    }

    /// Minimal parser scaffold: accepts newline-delimited `label,mime` entries.
    ///
    /// This keeps the repository dependency-free in restricted environments and is
    /// intended to be replaced by full `config.min.json` parsing.
    pub fn from_json_str(raw: &str) -> Result<Self, MagikaInferenceError> {
        let labels = raw
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let mut parts = line.splitn(2, ',');
                let label = parts.next().unwrap_or_default().trim().to_string();
                let mime = parts
                    .next()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty());
                LabelEntry {
                    label,
                    mime_type: mime,
                }
            })
            .filter(|entry| !entry.label.is_empty())
            .collect::<Vec<_>>();

        if labels.is_empty() {
            return Err(MagikaInferenceError::InvalidConfig(
                "no labels found in config input".to_string(),
            ));
        }

        Ok(Self {
            labels,
            metadata: HashMap::new(),
        })
    }
}
