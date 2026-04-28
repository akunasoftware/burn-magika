use std::path::Path;

use burn::tensor::backend::Backend;

use crate::{Error, FileType, MagikaModel};

pub struct Session<B: Backend> {
    model: MagikaModel<B>,
}

impl<B: Backend<FloatElem = f32>> Session<B> {
    pub fn new(device: &B::Device) -> Result<Self, Error> {
        let model = MagikaModel::<B>::from_embedded(device)?;
        Ok(Self { model })
    }

    pub fn from_file(device: &B::Device, path: impl AsRef<Path>) -> Result<Self, Error> {
        let model = MagikaModel::<B>::from_file(device, path)?;
        Ok(Self { model })
    }

    pub fn from_bytes(device: &B::Device, bytes: &[u8]) -> Result<Self, Error> {
        let model = MagikaModel::<B>::from_bytes(device, bytes)?;
        Ok(Self { model })
    }

    pub fn identify_file_sync(&mut self, path: impl AsRef<Path>) -> Result<FileType, Error> {
        self.model.identify_path(path)
    }

    pub async fn identify_file_async(&mut self, path: impl AsRef<Path>) -> Result<FileType, Error> {
        self.model.identify_path(path)
    }

    pub fn identify_content_sync(&mut self, bytes: &[u8]) -> Result<FileType, Error> {
        self.model.identify_bytes(bytes)
    }

    pub async fn identify_content_async(&mut self, bytes: &[u8]) -> Result<FileType, Error> {
        self.model.identify_bytes(bytes)
    }
}
