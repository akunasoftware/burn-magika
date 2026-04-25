#[derive(Debug, Clone, PartialEq)]
pub struct RankedAlternative {
    pub label: String,
    pub mime_type: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub label: String,
    pub mime_type: Option<String>,
    pub confidence: f32,
    pub alternatives: Vec<RankedAlternative>,
}
