use crate::{config::ModelConfig, vendor::content::ContentType};

pub(crate) enum PreparedInput {
    Features(Vec<i32>),
    Ruled(ContentType),
}

pub(crate) fn prepare_input(input: &[u8], config: &ModelConfig) -> PreparedInput {
    if input.is_empty() {
        return PreparedInput::Ruled(ContentType::Empty);
    }

    let first_block_len = config.block_size.min(input.len());
    let first_block = &input[..first_block_len];
    let buffer_size = config.block_size.min(input.len());
    let beg = strip_prefix(&input[..buffer_size]);
    let end = strip_suffix(&input[input.len() - buffer_size..]);

    let mut features = vec![config.padding_token; config.features_size()];
    let (beg_features, end_features) = features.split_at_mut(config.beg_size);
    copy_features(beg_features, beg, 0);
    copy_features(end_features, end, 1);

    if features[config.min_file_size_for_dl - 1] != config.padding_token {
        return PreparedInput::Features(features);
    }

    if std::str::from_utf8(first_block).is_ok() {
        PreparedInput::Ruled(ContentType::Txt)
    } else {
        PreparedInput::Ruled(ContentType::Unknown)
    }
}

fn copy_features(dst: &mut [i32], src: &[u8], align: usize) {
    let dst_len = dst.len();
    let len = dst_len.min(src.len());
    let dst_start = (dst_len - len) * align;
    let src_start = (src.len() - len) * align;

    for index in 0..len {
        dst[dst_start + index] = src[src_start + index] as i32;
    }
}

fn strip_prefix(xs: &[u8]) -> &[u8] {
    strip(xs, |slice| slice.split_first())
}

fn strip_suffix(xs: &[u8]) -> &[u8] {
    strip(xs, |slice| slice.split_last())
}

fn strip<'a>(
    mut xs: &'a [u8],
    mut split: impl FnMut(&'a [u8]) -> Option<(&'a u8, &'a [u8])>,
) -> &'a [u8] {
    while let Some((&x, ys)) = split(xs) {
        if !is_whitespace(x) {
            break;
        }
        xs = ys;
    }

    xs
}

fn is_whitespace(x: u8) -> bool {
    x.is_ascii_whitespace() || x == 0x0b
}

/// Legacy helper kept for callers that still want a normalized head/tail feature vector.
pub fn preprocess_bytes(
    input: &[u8],
    head_len: usize,
    tail_len: usize,
    feature_len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; feature_len];
    if feature_len == 0 {
        return out;
    }

    let mut write_idx = 0;

    for &b in input.iter().take(head_len) {
        if write_idx >= feature_len {
            return out;
        }
        out[write_idx] = f32::from(b) / 255.0;
        write_idx += 1;
    }

    let tail_take = tail_len.min(input.len());
    let tail_start = input.len().saturating_sub(tail_take);
    for &b in &input[tail_start..] {
        if write_idx >= feature_len {
            break;
        }
        out[write_idx] = f32::from(b) / 255.0;
        write_idx += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use crate::{config::runtime_config, vendor::content::ContentType};

    use super::{PreparedInput, prepare_input, preprocess_bytes};

    #[test]
    fn pads_and_normalizes() {
        let input = [0_u8, 255_u8, 128_u8];
        let v = preprocess_bytes(&input[..], 2, 2, 6);
        assert_eq!(v.len(), 6);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 1.0);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn short_utf8_input_is_ruled_as_text() {
        let config = runtime_config();
        match prepare_input(b"hello".as_slice(), &config) {
            PreparedInput::Ruled(ContentType::Txt) => {}
            _ => panic!("expected ruled text"),
        }
    }
}
