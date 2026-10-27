//! Verified downloads: streamed to a `.part` file, hashed on the fly, checked against the size and
//! digest the publisher lists, then moved into place. Hugging Face lists both for every file
//! (SHA-256 for LFS files); the RVM bucket publishes an MD5.

use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::Engine;
use corridorkey_core::{ProgressEvent, ProgressSink};
use md5::Md5;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Time allowed to connect; transfers themselves have no limit (the checkpoint is 2.7 GB).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Read buffer of a transfer or a hash.
const CHUNK_BYTES: usize = 1 << 20;
const BYTES_PER_MB: u64 = 1_000_000;

/// What a downloaded file must match.
#[derive(Debug, Clone, Default)]
pub struct Expected {
    pub size: Option<u64>,
    pub sha256: Option<String>,
    pub md5_b64: Option<String>,
}

/// A file of a Hugging Face repository at a revision.
#[derive(Debug, Clone)]
pub struct HubFile {
    pub path: String,
    pub expected: Expected,
}

#[derive(Deserialize)]
struct RepoInfo {
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    sha256: String,
    size: u64,
}

pub fn http_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(concat!("corridorkey-cli/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(None)
        .build()
        .context("failed to build HTTP client")
}

/// Files of `repo` at `revision`, with their sizes and (LFS files) SHA-256.
pub fn hub_files(client: &reqwest::blocking::Client, repo: &str, revision: &str) -> Result<Vec<HubFile>> {
    let url = format!("https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true");
    let info: RepoInfo = client
        .get(&url)
        .send()
        .and_then(|r| r.error_for_status())
        .with_context(|| format!("listing {repo}@{revision}"))?
        .json()
        .with_context(|| format!("reading the file list of {repo}@{revision}"))?;
    Ok(info
        .siblings
        .into_iter()
        .map(|s| HubFile {
            expected: Expected {
                size: s.lfs.as_ref().map(|l| l.size).or(s.size),
                sha256: s.lfs.map(|l| l.sha256),
                md5_b64: None,
            },
            path: s.rfilename,
        })
        .collect())
}

/// URL of `path` in `repo` at `revision`.
pub fn hub_url(repo: &str, revision: &str, path: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{path}")
}

/// Whether `dest` already holds the expected file. Without `hash`, only the size is compared
/// (fast re-runs); with it, the digest too.
pub fn is_present(dest: &Path, expected: &Expected, hash: bool) -> Result<bool> {
    let Ok(meta) = std::fs::metadata(dest) else { return Ok(false) };
    if expected.size.is_some_and(|s| s != meta.len()) {
        return Ok(false);
    }
    if !hash && expected.size.is_some() {
        return Ok(true);
    }
    let (sha256, md5) = hash_file(dest)?;
    Ok(digests_match(expected, &sha256, &md5))
}

fn digests_match(expected: &Expected, sha256: &str, md5_b64: &str) -> bool {
    expected.sha256.as_ref().is_none_or(|e| e.eq_ignore_ascii_case(sha256))
        && expected.md5_b64.as_ref().is_none_or(|e| e == md5_b64)
}

/// `(hex SHA-256, base64 MD5)` of a file.
pub fn hash_file(path: &Path) -> Result<(String, String)> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let (mut sha, mut md5) = (Sha256::new(), Md5::new());
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
        md5.update(&buf[..n]);
    }
    Ok((hex::encode(sha.finalize()), base64::engine::general_purpose::STANDARD.encode(md5.finalize())))
}

/// Downloads `url` to `dest` and checks it against `expected` before moving it into place.
/// Reports MB transferred as stage `(step, steps, name)`.
pub fn download(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    expected: &Expected,
    (step, steps, name): (usize, usize, &str),
    progress: &dyn ProgressSink,
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut resp = client.get(url).send().and_then(|r| r.error_for_status()).with_context(|| format!("GET {url}"))?;
    let total = expected.size.or(resp.content_length()).unwrap_or(0);
    let tmp = dest.with_file_name(format!("{}.part", dest.file_name().unwrap_or_default().to_string_lossy()));
    let mut out = std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let (mut sha, mut md5) = (Sha256::new(), Md5::new());
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut written = 0u64;
    let report = |done: u64| {
        progress.report(&ProgressEvent {
            stage: name,
            step,
            steps,
            done: done / BYTES_PER_MB,
            total: total.div_ceil(BYTES_PER_MB),
            unit: "MB",
        })
    };
    loop {
        let n = resp.read(&mut buf).with_context(|| format!("reading {url}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        sha.update(&buf[..n]);
        md5.update(&buf[..n]);
        written += n as u64;
        report(written);
    }
    out.flush()?;
    drop(out);
    // Completes the bar even when the size was rounded up
    report(total.div_ceil(BYTES_PER_MB) * BYTES_PER_MB);

    let fail = |msg: String| -> Result<()> {
        let _ = std::fs::remove_file(&tmp);
        bail!(msg)
    };
    if expected.size.is_some_and(|s| s != written) {
        return fail(format!("size mismatch for {url}: expected {} bytes, got {written}", expected.size.unwrap_or(0)));
    }
    let (sha256, md5_b64) = (hex::encode(sha.finalize()), base64::engine::general_purpose::STANDARD.encode(md5.finalize()));
    if !digests_match(expected, &sha256, &md5_b64) {
        return fail(format!("checksum mismatch for {url}"));
    }
    std::fs::rename(&tmp, dest).with_context(|| format!("moving {} to {}", tmp.display(), dest.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digests_are_checked_when_listed() {
        let dir = std::env::temp_dir().join(format!("corridorkey-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.bin");
        std::fs::write(&path, b"corridorkey").unwrap();
        let (sha, md5) = hash_file(&path).unwrap();
        let good = Expected { size: Some(11), sha256: Some(sha.to_uppercase()), md5_b64: Some(md5) };
        assert!(is_present(&path, &good, true).unwrap());
        let bad = Expected { sha256: Some("00".repeat(32)), ..good.clone() };
        assert!(is_present(&path, &bad, false).unwrap(), "size only without hashing");
        assert!(!is_present(&path, &bad, true).unwrap());
        assert!(!is_present(&path, &Expected { size: Some(12), ..good }, false).unwrap());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
