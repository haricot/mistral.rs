//! B2-C: Voxtral4B GGUF Tekken tokenizer as executed by CrispASR voxtral4b.cpp.
//! Its UTF-8 pre-split is simpler than the full Tekken regex. A C++/native
//! tokenizer parity check remains mandatory before any runtime promotion.
use std::collections::{HashMap, HashSet};

use anyhow::{bail, ensure, Context, Result};
use candle_core::quantized::gguf_file::Value;
use mistralrs_quant::GgufArchive;

#[derive(Debug)]
pub(crate) struct Voxtral4Tekken {
    specials: Vec<String>,
    ranks: HashMap<Vec<u8>, u32>,
    active: usize,
    inactive: usize,
    embedding_rows: usize,
}

impl Voxtral4Tekken {
    pub(crate) fn from_archive(archive: &GgufArchive) -> Result<Self> {
        super::voxtral4::build_voxtral4_bindings(archive)?;
        let specials = match archive.metadata_value("tokenizer.tekken.specials") {
            Some(Value::Array(values)) => values
                .iter()
                .map(|v| match v {
                    Value::String(s) => Ok(s.clone()),
                    _ => bail!("Tekken special token is not a string"),
                })
                .collect::<Result<Vec<_>>>()?,
            _ => bail!("Voxtral GGUF lacks tokenizer.tekken.specials"),
        };
        ensure!(
            specials.len() == count(archive, "tokenizer.tekken.n_specials")?,
            "Tekken special count mismatch"
        );
        let vocab = count(archive, "tokenizer.tekken.n_vocab")?;
        let rows = count(archive, "voxtral4b.llm.vocab_size")?;
        ensure!(rows == 131_072, "unqualified Voxtral4b embedding count");
        let audio_id = count(archive, "voxtral4b.audio_token_id")?;
        ensure!(
            specials.get(audio_id).map(String::as_str) == Some("[AUDIO]"),
            "Tekken audio token must be [AUDIO] at its declared ID"
        );
        let tensor = archive.tensor_data("tokenizer.tekken.vocab_tensor")?;
        ensure!(
            tensor.info().dtype().raw() == 0,
            "Tekken byte payload must be F32"
        );
        let data = tensor.bytes();
        ensure!(
            data.len() % 4 == 0 && data.len() <= 8_000_000,
            "invalid Tekken F32 payload length"
        );
        let mut bytes = Vec::with_capacity(data.len() / 4);
        for four in data.chunks_exact(4) {
            let value = f32::from_le_bytes(four.try_into().expect("fixed four-byte chunks"));
            ensure!(
                value.is_finite() && (0.0..=255.0).contains(&value) && value.fract() == 0.0,
                "Tekken F32 vocabulary contains a non-byte value"
            );
            bytes.push(value as u8);
        }
        Self::from_packed(&bytes, specials, vocab, rows)
    }

    pub(crate) fn from_packed(
        packed: &[u8],
        specials: Vec<String>,
        serialized: usize,
        embedding_rows: usize,
    ) -> Result<Self> {
        ensure!(
            !specials.is_empty()
                && specials.len() < embedding_rows
                && (1..=200_000).contains(&serialized),
            "invalid Tekken rank/special/embedding counts"
        );
        let mut specials_seen = HashSet::new();
        ensure!(
            specials
                .iter()
                .all(|s| !s.is_empty() && specials_seen.insert(s)),
            "duplicate or empty Tekken special token"
        );
        let active = serialized.min(embedding_rows - specials.len());
        let inactive = serialized - active;
        let mut ranks = HashMap::with_capacity(active);
        let mut pos = 0usize;
        for rank in 0..serialized {
            ensure!(pos + 2 <= packed.len(), "truncated Tekken BPE rank length");
            let len = u16::from_le_bytes([packed[pos], packed[pos + 1]]) as usize;
            pos += 2;
            let end = pos.checked_add(len).context("Tekken BPE rank overflow")?;
            ensure!(end <= packed.len(), "truncated Tekken BPE token bytes");
            if rank < active {
                ensure!(
                    ranks
                        .insert(packed[pos..end].to_vec(), rank as u32)
                        .is_none(),
                    "duplicate active Tekken BPE token"
                );
            }
            pos = end;
        }
        ensure!(
            pos == packed.len(),
            "trailing bytes in Tekken BPE vocabulary"
        );
        ensure!(
            (0u8..=255).all(|b| ranks.contains_key(&[b][..])),
            "Tekken active vocabulary does not cover all 256 single bytes"
        );
        Ok(Self {
            specials,
            ranks,
            active,
            inactive,
            embedding_rows,
        })
    }

    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.active, self.inactive)
    }

    pub(crate) fn encode_crispasr(&self, text: &str) -> Result<Vec<u32>> {
        ensure!(text.len() <= 1_000_000, "Tekken input exceeds byte bound");
        let mut ids = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            if let Some((id, special)) = self
                .specials
                .iter()
                .enumerate()
                .find(|(_, special)| text[pos..].starts_with(special.as_str()))
            {
                ids.push(id as u32);
                pos += special.len();
                continue;
            }
            let next = self
                .specials
                .iter()
                .filter_map(|special| text[pos..].find(special).map(|i| pos + i))
                .min()
                .unwrap_or(text.len());
            ensure!(next > pos, "Tekken special scanner made no progress");
            for part in crispasr4b_split(&text[pos..next]) {
                self.merge_ranked(part.as_bytes(), &mut ids);
            }
            pos = next;
        }
        ensure!(
            ids.iter().all(|id| (*id as usize) < self.embedding_rows),
            "Tekken emitted an out-of-bounds embedding ID"
        );
        Ok(ids)
    }

    fn merge_ranked(&self, bytes: &[u8], ids: &mut Vec<u32>) {
        if bytes.is_empty() {
            return;
        }
        let mut pieces: Vec<(usize, usize)> = (0..bytes.len()).map(|i| (i, 1)).collect();
        while pieces.len() > 1 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..pieces.len() - 1 {
                let (start, len) = pieces[i];
                if let Some(&rank) = self.ranks.get(&bytes[start..start + len + pieces[i + 1].1]) {
                    if best.is_none_or(|(_, prior)| rank < prior) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((index, _)) = best else {
                break;
            };
            pieces[index].1 += pieces[index + 1].1;
            pieces.remove(index + 1);
        }
        for (start, len) in pieces {
            ids.push(
                self.ranks
                    .get(&bytes[start..start + len])
                    .map_or(0, |rank| *rank + self.specials.len() as u32),
            );
        }
    }
}

fn count(archive: &GgufArchive, key: &str) -> Result<usize> {
    match archive.metadata_value(key) {
        Some(Value::U32(value)) => Ok(*value as usize),
        Some(Value::U64(value)) => Ok(usize::try_from(*value)?),
        _ => bail!("Voxtral GGUF lacks unsigned metadata {key}"),
    }
}

/// Mirrors the simple byte-class pre-split in CrispASR src/voxtral4b.cpp.
/// It is deliberately not advertised as complete Tekken Unicode regex parity.
fn crispasr4b_split(input: &str) -> Vec<&str> {
    let bytes = input.as_bytes();
    let mut parts = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let first = bytes[i];
        if matches!(first, b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
            while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
                i += 1;
            }
        } else if first.is_ascii_alphanumeric() {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
        } else if first >= 0x80 {
            i += 1;
            while i < bytes.len() && (bytes[i] & 0xc0) == 0x80 {
                i += 1;
            }
            while i < bytes.len() && bytes[i] >= 0x80 {
                i += 1;
                while i < bytes.len() && (bytes[i] & 0xc0) == 0x80 {
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
        parts.push(&input[start..i]);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Voxtral4Tekken {
        let specials = vec!["<unk>".to_string(), "[AUDIO]".to_string()];
        let mut tokens = (0u8..=255).map(|b| vec![b]).collect::<Vec<_>>();
        tokens.extend([
            b"Hi".to_vec(),
            b"Bo".to_vec(),
            b"Bon".to_vec(),
            b"jo".to_vec(),
            b"jou".to_vec(),
            b"jour".to_vec(),
            b"Bonjour".to_vec(),
            b"inert".to_vec(),
        ]);
        let mut packed = Vec::new();
        for token in &tokens {
            packed.extend_from_slice(&(token.len() as u16).to_le_bytes());
            packed.extend_from_slice(token);
        }
        Voxtral4Tekken::from_packed(&packed, specials, tokens.len(), 265).unwrap()
    }

    #[test]
    fn crispasr_utf8_pretoken_splits() {
        assert_eq!(
            crispasr4b_split("Bonjour à l'école!  42\n[AUDIO]"),
            [
                "Bonjour", " ", "à", " ", "l", "'", "é", "cole", "!", "  ", "42", "\n", "[",
                "AUDIO", "]"
            ]
        );
        assert_eq!(crispasr4b_split("texte—café"), ["texte", "—", "caf", "é"]);
    }

    #[test]
    fn bpe_merges_and_reserved_audio_token() -> Result<()> {
        let tok = fixture();
        assert_eq!(tok.counts(), (263, 1));
        assert_eq!(tok.encode_crispasr("Hi Bonjour[AUDIO]")?, [258, 34, 264, 1]);
        assert!(tok.encode_crispasr("")?.is_empty());
        Ok(())
    }

    #[test]
    fn corrupt_rank_payload_rejected() {
        let mut data = Vec::new();
        for b in 0u8..=255 {
            data.extend_from_slice(&1u16.to_le_bytes());
            data.push(b);
        }
        data.extend_from_slice(&3u16.to_le_bytes());
        data.push(b'x');
        assert!(Voxtral4Tekken::from_packed(&data, vec!["<unk>".into()], 257, 400).is_err());
    }

    /// Optional real-model differential: C++ oracle linked to the verified
    /// CrispASR header and the Rust GGUF tokenizer must agree ID by ID.
    /// The Python driver sets all three environment variables, including the
    /// explicitly fingerprinted 150000-rank Tekken sidecar.
    #[test]
    fn optional_crispasr_cpp_differential() -> Result<()> {
        let Ok(model) = std::env::var("VOXTRAL4_GGUF_PATH") else {
            return Ok(());
        };
        let Ok(oracle) = std::env::var("VOXTRAL4_CPP_ORACLE") else {
            return Ok(());
        };
        let Ok(sidecar) = std::env::var("VOXTRAL4_CPP_SIDECAR") else {
            return Ok(());
        };
        let archive = GgufArchive::open_file(model)?;
        let tokenizer = Voxtral4Tekken::from_archive(&archive)?;
        let corpus: Vec<String> = serde_json::from_str(include_str!(
            "../../../tools/voxtral4_gguf_french_corpus.json"
        ))?;
        let mut command = std::process::Command::new(&oracle);
        command.arg(&sidecar).arg("runtime");
        for phrase in &corpus {
            use std::fmt::Write;
            let mut encoded = String::with_capacity(phrase.len() * 2);
            for byte in phrase.as_bytes() {
                write!(&mut encoded, "{byte:02x}")?;
            }
            command.arg(encoded);
        }
        let output = command.output().context("CrispASR oracle did not launch")?;
        ensure!(
            output.status.success(),
            "CrispASR reference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = std::str::from_utf8(&output.stdout)?;
        let actual_lines = stdout.lines().collect::<Vec<_>>();
        ensure!(
            actual_lines.len() == corpus.len(),
            "CrispASR oracle returned {} responses for {} phrases",
            actual_lines.len(),
            corpus.len()
        );
        for (index, (phrase, row)) in corpus.iter().zip(actual_lines).enumerate() {
            let expected: Vec<u32> = serde_json::from_str(row)?;
            let actual = tokenizer.encode_crispasr(phrase)?;
            ensure!(
                actual == expected,
                "CrispASR Rust token mismatch at case {index} {phrase:?}: native={actual:?} reference={expected:?}"
            );
        }
        println!("B2C2_CPP_RUST_RUNTIME_PARITY=PASS cases={}", corpus.len());
        Ok(())
    }

    #[test]
    fn optional_real_gguf_tok_ids() -> Result<()> {
        let Ok(path) = std::env::var("VOXTRAL4_GGUF_PATH") else {
            return Ok(());
        };
        let archive = GgufArchive::open_file(path)?;
        let tok = Voxtral4Tekken::from_archive(&archive)?;
        assert_eq!(tok.counts(), (130_072, 19_928));
        assert_eq!(tok.encode_crispasr("[AUDIO]")?, [24]);
        Ok(())
    }
}
