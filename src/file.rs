pub struct TypeInfo {
    pub label: &'static str,
    pub mime_type: &'static str,
    pub group: &'static str,
    pub description: &'static str,
    pub extensions: &'static [&'static str],
    pub is_text: bool,
}
