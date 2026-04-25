use std::borrow::Cow;

use crate::vendor::{content::ContentType, model as vendor_model};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub beg_size: usize,
    pub end_size: usize,
    pub min_file_size_for_dl: usize,
    pub padding_token: i32,
    pub block_size: usize,
    pub thresholds: Cow<'static, [f32; ContentType::SIZE]>,
    pub overwrite_map: Cow<'static, [ContentType; ContentType::SIZE]>,
}

impl ModelConfig {
    pub(crate) fn features_size(&self) -> usize {
        self.beg_size + self.end_size
    }
}

pub(crate) fn runtime_config() -> ModelConfig {
    vendor_model::CONFIG.clone()
}
