use candle_core::Result;
use half::f16;

pub const HQZ4_SCHEMA_VERSION: u32 = 1;
pub const HQZ4_BITS: usize = 4;
pub const HQZ4_DEFAULT_GROUP_SIZE: usize = 128;
pub const HQZ4_LAYOUT_ROW_MAJOR_NIBBLES: u8 = 0;
pub const HQZ4_TRANSFORM_SHARED_RHT: u8 = 0;

const HQZ4_MAX_LEVEL: i8 = 7;
const HQZ4_NIBBLE_MASK: u8 = 0x0f;
const SPLITMIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MULTIPLIER_1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MULTIPLIER_2: u64 = 0x94d0_49bb_1331_11eb;
const GROUP_MIX: u64 = 0xd1b5_4a32_d192_ed03;
const ELEMENT_MIX: u64 = 0x8cb9_2baa_7f3d_d15b;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hqz4Config {
    pub group_size: usize,
    pub seed: u64,
}

impl Default for Hqz4Config {
    fn default() -> Self {
        Self {
            group_size: HQZ4_DEFAULT_GROUP_SIZE,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hqz4Tensor {
    rows: usize,
    cols: usize,
    group_size: usize,
    seed: u64,
    scales: Vec<f16>,
    codes: Vec<u8>,
}

impl Hqz4Tensor {
    pub fn encode(weights: &[f32], rows: usize, cols: usize, cfg: Hqz4Config) -> Result<Self> {
        validate_layout(rows, cols, cfg.group_size)?;
        let elements = checked_elements(rows, cols)?;
        if weights.len() != elements {
            candle_core::bail!(
                "HQZ4 weight length {} does not match shape [{rows}, {cols}].",
                weights.len()
            );
        }
        if weights.iter().any(|value| !value.is_finite()) {
            candle_core::bail!("HQZ4 weights must be finite.");
        }

        let groups_per_row = cols / cfg.group_size;
        let scale_count = rows
            .checked_mul(groups_per_row)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 scale count overflow.".into()))?;
        let mut scales = Vec::with_capacity(scale_count);
        let mut codes = vec![0u8; elements / 2];
        let mut rotated = vec![0f32; cfg.group_size];

        for row in 0..rows {
            for group in 0..groups_per_row {
                let start = row * cols + group * cfg.group_size;
                rotated.copy_from_slice(&weights[start..start + cfg.group_size]);
                rotate_weight_group(&mut rotated, cfg.seed, group);

                let max_abs = rotated
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0f32, f32::max);
                let scale = if max_abs == 0.0 {
                    f16::ZERO
                } else {
                    let scale = f16::from_f32(max_abs / HQZ4_MAX_LEVEL as f32);
                    if !scale.is_finite() || scale == f16::ZERO {
                        candle_core::bail!(
                            "HQZ4 group scale is outside the finite F16 range at row {row}, group {group}."
                        );
                    }
                    scale
                };
                scales.push(scale);

                let scale_f32 = scale.to_f32();
                for (offset, value) in rotated.iter().copied().enumerate() {
                    let quantized = if scale == f16::ZERO {
                        0
                    } else {
                        (value / scale_f32)
                            .round()
                            .clamp(-(HQZ4_MAX_LEVEL as f32), HQZ4_MAX_LEVEL as f32)
                            as i8
                    };
                    write_code(&mut codes, start + offset, quantized);
                }
            }
        }

        Self::from_parts(rows, cols, cfg, scales, codes)
    }

    pub fn from_parts(
        rows: usize,
        cols: usize,
        cfg: Hqz4Config,
        scales: Vec<f16>,
        codes: Vec<u8>,
    ) -> Result<Self> {
        validate_layout(rows, cols, cfg.group_size)?;
        let elements = checked_elements(rows, cols)?;
        let expected_scales = rows
            .checked_mul(cols / cfg.group_size)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 scale count overflow.".into()))?;
        if scales.len() != expected_scales {
            candle_core::bail!(
                "HQZ4 scale count {} does not match expected count {expected_scales}.",
                scales.len()
            );
        }
        if scales.iter().any(|scale| !scale.is_finite() || *scale < f16::ZERO) {
            candle_core::bail!("HQZ4 scales must be finite and non-negative.");
        }
        let expected_codes = elements / 2;
        if codes.len() != expected_codes {
            candle_core::bail!(
                "HQZ4 code length {} does not match expected length {expected_codes}.",
                codes.len()
            );
        }
        if (0..elements).any(|index| read_code(&codes, index) < -HQZ4_MAX_LEVEL) {
            candle_core::bail!("HQZ4 code payload contains the reserved -8 level.");
        }
        for (group_index, scale) in scales.iter().enumerate() {
            if *scale != f16::ZERO {
                continue;
            }
            let row = group_index / (cols / cfg.group_size);
            let group = group_index % (cols / cfg.group_size);
            let start = row * cols + group * cfg.group_size;
            if (start..start + cfg.group_size).any(|index| read_code(&codes, index) != 0) {
                candle_core::bail!("HQZ4 zero-scale group contains non-zero codes.");
            }
        }

        Ok(Self {
            rows,
            cols,
            group_size: cfg.group_size,
            seed: cfg.seed,
            scales,
            codes,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn scales(&self) -> &[f16] {
        &self.scales
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn decode_rotated_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.rows {
            candle_core::bail!("HQZ4 row {row} is outside row count {}.", self.rows);
        }
        let groups_per_row = self.cols / self.group_size;
        let mut output = vec![0f32; self.cols];
        for group in 0..groups_per_row {
            let scale = self.scales[row * groups_per_row + group].to_f32();
            let start = row * self.cols + group * self.group_size;
            let output_start = group * self.group_size;
            for offset in 0..self.group_size {
                output[output_start + offset] =
                    read_code(&self.codes, start + offset) as f32 * scale;
            }
        }
        Ok(output)
    }

    pub fn decode_row(&self, row: usize) -> Result<Vec<f32>> {
        let mut output = self.decode_rotated_row(row)?;
        for group in 0..self.cols / self.group_size {
            let start = group * self.group_size;
            inverse_rotate_weight_group(
                &mut output[start..start + self.group_size],
                self.seed,
                group,
            );
        }
        Ok(output)
    }

    pub fn decode(&self) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(checked_elements(self.rows, self.cols)?);
        for row in 0..self.rows {
            output.extend(self.decode_row(row)?);
        }
        Ok(output)
    }

    pub fn transform_activation(&self, activation: &[f32]) -> Result<Vec<f32>> {
        if activation.len() != self.cols {
            candle_core::bail!(
                "HQZ4 activation length {} does not match input width {}.",
                activation.len(),
                self.cols
            );
        }
        if activation.iter().any(|value| !value.is_finite()) {
            candle_core::bail!("HQZ4 activations must be finite.");
        }

        let mut output = activation.to_vec();
        for group in 0..self.cols / self.group_size {
            let start = group * self.group_size;
            transform_activation_group(
                &mut output[start..start + self.group_size],
                self.seed,
                group,
            );
        }
        Ok(output)
    }

    pub fn matvec(&self, activation: &[f32]) -> Result<Vec<f32>> {
        let activation = self.transform_activation(activation)?;
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let weights = self.decode_rotated_row(row)?;
            output.push(
                weights
                    .iter()
                    .zip(&activation)
                    .map(|(weight, activation)| weight * activation)
                    .sum(),
            );
        }
        Ok(output)
    }
}

fn validate_layout(rows: usize, cols: usize, group_size: usize) -> Result<()> {
    if rows == 0 || cols == 0 {
        candle_core::bail!("HQZ4 matrices must have non-zero dimensions.");
    }
    if group_size < 2 || !group_size.is_power_of_two() {
        candle_core::bail!("HQZ4 group size {group_size} must be a power of two of at least 2.");
    }
    if !cols.is_multiple_of(group_size) {
        candle_core::bail!(
            "HQZ4 input width {cols} must be divisible by group size {group_size}."
        );
    }
    checked_elements(rows, cols)?;
    Ok(())
}

fn checked_elements(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| candle_core::Error::Msg("HQZ4 element count overflow.".into()))
}

fn rotate_weight_group(values: &mut [f32], seed: u64, group: usize) {
    apply_signs(values, seed, group);
    normalized_hadamard(values);
}

fn inverse_rotate_weight_group(values: &mut [f32], seed: u64, group: usize) {
    normalized_hadamard(values);
    apply_signs(values, seed, group);
}

fn transform_activation_group(values: &mut [f32], seed: u64, group: usize) {
    apply_signs(values, seed, group);
    normalized_hadamard(values);
}

fn normalized_hadamard(values: &mut [f32]) {
    let mut half = 1;
    while half < values.len() {
        let span = half * 2;
        for start in (0..values.len()).step_by(span) {
            for offset in 0..half {
                let left = values[start + offset];
                let right = values[start + offset + half];
                values[start + offset] = left + right;
                values[start + offset + half] = left - right;
            }
        }
        half = span;
    }
    let normalization = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value *= normalization;
    }
}

fn apply_signs(values: &mut [f32], seed: u64, group: usize) {
    for (index, value) in values.iter_mut().enumerate() {
        if sign_is_negative(seed, group, index) {
            *value = -*value;
        }
    }
}

fn sign_is_negative(seed: u64, group: usize, index: usize) -> bool {
    let state = seed
        ^ (group as u64).wrapping_mul(GROUP_MIX)
        ^ (index as u64).wrapping_mul(ELEMENT_MIX);
    splitmix64(state) & 1 != 0
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(SPLITMIX_INCREMENT);
    state = (state ^ (state >> 30)).wrapping_mul(SPLITMIX_MULTIPLIER_1);
    state = (state ^ (state >> 27)).wrapping_mul(SPLITMIX_MULTIPLIER_2);
    state ^ (state >> 31)
}

fn write_code(codes: &mut [u8], index: usize, value: i8) {
    let nibble = value as u8 & HQZ4_NIBBLE_MASK;
    let byte = &mut codes[index / 2];
    if index.is_multiple_of(2) {
        *byte = (*byte & 0xf0) | nibble;
    } else {
        *byte = (*byte & HQZ4_NIBBLE_MASK) | (nibble << 4);
    }
}

fn read_code(codes: &[u8], index: usize) -> i8 {
    let byte = codes[index / 2];
    let nibble = if index.is_multiple_of(2) {
        byte & HQZ4_NIBBLE_MASK
    } else {
        byte >> 4
    };
    (nibble << 4) as i8 >> 4
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ROWS: usize = 5;
    const TEST_COLS: usize = 128;
    const TEST_GROUP_SIZE: usize = 32;
    const TEST_SEED: u64 = 0x5eed_61;
    const TEST_MAX_RELATIVE_L2: f32 = 0.12;
    const TEST_MIN_COSINE: f32 = 0.995;

    fn test_weights() -> Vec<f32> {
        (0..TEST_ROWS * TEST_COLS)
            .map(|index| {
                let index = index as f32;
                (index * 0.037).sin() * 0.35 + (index * 0.011).cos() * 0.08
            })
            .collect()
    }

    fn test_config() -> Hqz4Config {
        Hqz4Config {
            group_size: TEST_GROUP_SIZE,
            seed: TEST_SEED,
        }
    }

    #[test]
    fn rht_round_trip_restores_values() {
        let mut values = (0..TEST_GROUP_SIZE)
            .map(|index| (index as f32 * 0.17).sin())
            .collect::<Vec<_>>();
        let expected = values.clone();
        rotate_weight_group(&mut values, TEST_SEED, 3);
        inverse_rotate_weight_group(&mut values, TEST_SEED, 3);

        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn encoding_is_deterministic() -> Result<()> {
        let weights = test_weights();
        let first = Hqz4Tensor::encode(&weights, TEST_ROWS, TEST_COLS, test_config())?;
        let second = Hqz4Tensor::encode(&weights, TEST_ROWS, TEST_COLS, test_config())?;
        let different_seed = Hqz4Tensor::encode(
            &weights,
            TEST_ROWS,
            TEST_COLS,
            Hqz4Config {
                group_size: TEST_GROUP_SIZE,
                seed: TEST_SEED + 1,
            },
        )?;

        assert_eq!(first, second);
        assert_ne!(first.codes(), different_seed.codes());
        assert_eq!(first.codes().len(), weights.len() / 2);
        assert_eq!(
            first.scales().len(),
            TEST_ROWS * TEST_COLS / TEST_GROUP_SIZE
        );
        Ok(())
    }

    #[test]
    fn signed_nibbles_are_low_first() -> Result<()> {
        let encoded = Hqz4Tensor::from_parts(
            1,
            2,
            Hqz4Config {
                group_size: 2,
                seed: 0,
            },
            vec![f16::from_f32(1.0)],
            vec![0x79],
        )?;

        assert_eq!(encoded.decode_rotated_row(0)?, vec![-7.0, 7.0]);
        Ok(())
    }

    #[test]
    fn row_decode_matches_full_decode() -> Result<()> {
        let encoded =
            Hqz4Tensor::encode(&test_weights(), TEST_ROWS, TEST_COLS, test_config())?;
        let decoded = encoded.decode()?;

        for row in 0..TEST_ROWS {
            assert_eq!(
                encoded.decode_row(row)?,
                decoded[row * TEST_COLS..(row + 1) * TEST_COLS]
            );
        }
        Ok(())
    }

    #[test]
    fn codec_preserves_weight_geometry() -> Result<()> {
        let weights = test_weights();
        let decoded =
            Hqz4Tensor::encode(&weights, TEST_ROWS, TEST_COLS, test_config())?.decode()?;
        let dot = weights
            .iter()
            .zip(&decoded)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let source_l2 = weights.iter().map(|value| value * value).sum::<f32>();
        let decoded_l2 = decoded.iter().map(|value| value * value).sum::<f32>();
        let error_l2 = weights
            .iter()
            .zip(&decoded)
            .map(|(left, right)| {
                let error = left - right;
                error * error
            })
            .sum::<f32>();
        let cosine = dot / (source_l2 * decoded_l2).sqrt();
        let relative_l2 = (error_l2 / source_l2).sqrt();

        assert!(cosine > TEST_MIN_COSINE, "cosine={cosine}");
        assert!(
            relative_l2 < TEST_MAX_RELATIVE_L2,
            "relative_l2={relative_l2}"
        );
        Ok(())
    }

    #[test]
    fn rotated_matvec_matches_decoded_dense_matvec() -> Result<()> {
        let encoded =
            Hqz4Tensor::encode(&test_weights(), TEST_ROWS, TEST_COLS, test_config())?;
        let activation = (0..TEST_COLS)
            .map(|index| (index as f32 * 0.023).cos() * 0.5)
            .collect::<Vec<_>>();
        let actual = encoded.matvec(&activation)?;
        let decoded = encoded.decode()?;
        let expected = decoded
            .chunks_exact(TEST_COLS)
            .map(|row| {
                row.iter()
                    .zip(&activation)
                    .map(|(weight, activation)| weight * activation)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();

        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 1e-4 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance);
        }
        Ok(())
    }

    #[test]
    fn from_parts_rejects_reserved_code() {
        let error = Hqz4Tensor::from_parts(
            1,
            2,
            Hqz4Config {
                group_size: 2,
                seed: 0,
            },
            vec![f16::from_f32(1.0)],
            vec![0x08],
        )
        .unwrap_err();
        assert!(error.to_string().contains("reserved -8"));
    }

    #[test]
    fn encode_rejects_invalid_inputs() {
        let invalid_group = Hqz4Tensor::encode(
            &[0.0; 12],
            2,
            6,
            Hqz4Config {
                group_size: 3,
                seed: 0,
            },
        )
        .unwrap_err();
        assert!(invalid_group.to_string().contains("power of two"));

        let non_finite = Hqz4Tensor::encode(
            &[0.0, f32::NAN],
            1,
            2,
            Hqz4Config {
                group_size: 2,
                seed: 0,
            },
        )
        .unwrap_err();
        assert!(non_finite.to_string().contains("finite"));
    }
}
