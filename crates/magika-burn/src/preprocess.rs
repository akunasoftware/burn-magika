/// Magika-style byte slicing scaffold.
///
/// The function keeps deterministic head/tail windows and zero pads to `feature_len`.
/// This is designed to be replaced with exact upstream-equivalent preprocessing once
/// config fields are mapped one-to-one with Magika's official values.
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
    use super::preprocess_bytes;

    #[test]
    fn pads_and_normalizes() {
        let v = preprocess_bytes(&[0, 255, 128], 2, 2, 6);
        assert_eq!(v.len(), 6);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 1.0);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }
}
