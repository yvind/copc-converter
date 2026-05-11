/// Validate consistency of scanned input files before building the octree.
use crate::Error;
use crate::extra_bytes::{ParsedExtraBytes, diff_schemas, merge_stats_into_canonical};
use crate::octree::{ScanResult, input_to_copc_format};
use std::path::PathBuf;
use tracing::{debug, warn};

/// Validated output: consistent properties across all input files.
#[derive(Debug)]
pub struct ValidatedInputs {
    /// WKT CRS string from input files (if present). Single canonical copy
    /// from the first scanned file — per-file WKT bytes are discarded
    /// after hashing during scan to keep memory O(1) in file count.
    pub wkt_crs: Option<Vec<u8>>,
    /// LAS Extra Bytes VLR payload (if present). Single canonical copy
    /// from the first scanned file, with per-field min/max stats merged
    /// across every input so the output COPC honestly advertises the
    /// full range of merged data. `None` when no input declares extras.
    pub extra_bytes_vlr: Option<Vec<u8>>,
    /// Trailing extra-byte width per point record. Zero when no extras
    /// are declared. Validated to be uniform across all inputs.
    pub num_extra_bytes: u16,
    /// COPC output point format (6, 7, or 8).
    pub point_format: u8,
}

/// Returns true if the LAS point format includes GPS time.
fn format_has_gps_time(fmt: u8) -> bool {
    // LAS formats 0 and 2 lack GPS time; all others (1, 3–10) include it.
    !matches!(fmt, 0 | 2)
}

/// Check that all scanned files agree on CRS and point format,
/// and derive the COPC output point format.
pub fn validate(
    input_files: &[PathBuf],
    results: &[ScanResult],
    canonical_wkt: Option<Vec<u8>>,
    canonical_extra_bytes_vlr: Option<Vec<u8>>,
    temporal_index: bool,
) -> crate::Result<ValidatedInputs> {
    let first = &results[0];
    let wkt_hash = first.wkt_crs_hash;
    let first_format = first.point_format_id;

    // Reference file for Extra Bytes diagnostics: the first scanned
    // file that declares any extras. Used so error messages and the
    // canonical schema both reference the file the user can grep for.
    let reference_extras_idx = results
        .iter()
        .position(|r| r.extra_bytes_parsed.is_some())
        .unwrap_or(0);
    let reference_extras_hash = results[reference_extras_idx].extra_bytes_schema_hash;
    let reference_num_extras = results[reference_extras_idx].num_extra_bytes;
    let reference_parsed = results[reference_extras_idx].extra_bytes_parsed.as_ref();

    for (i, r) in results.iter().enumerate() {
        if i == 0 {
            continue;
        }
        if r.wkt_crs_hash != wkt_hash {
            return Err(Error::CrsMismatch {
                file_a: input_files[0].clone(),
                file_b: input_files[i].clone(),
            });
        }
        if r.point_format_id != first_format {
            return Err(Error::PointFormatMismatch {
                file_a: input_files[0].clone(),
                format_a: first_format,
                file_b: input_files[i].clone(),
                format_b: r.point_format_id,
            });
        }
    }

    // Separate pass for Extra Bytes — compares each file against the
    // reference (first file with extras), not file 0, so error messages
    // name the file that actually defines the canonical schema.
    for (i, r) in results.iter().enumerate() {
        if i == reference_extras_idx {
            continue;
        }
        if r.extra_bytes_schema_hash == reference_extras_hash
            && r.num_extra_bytes == reference_num_extras
        {
            continue;
        }
        return Err(extras_mismatch_error(
            input_files,
            reference_extras_idx,
            reference_parsed,
            reference_num_extras,
            i,
            &results[i],
        ));
    }

    if temporal_index && !format_has_gps_time(first_format) {
        return Err(Error::NoGpsTime {
            format: first_format,
        });
    }

    // Stat-merge: take the canonical VLR bytes (first file's payload)
    // and patch the per-field min/max stats with the union across all
    // inputs. Producers compute per-file stats from each tile's own
    // points; without this, the output COPC would advertise stats from
    // an arbitrary single tile.
    let extra_bytes_vlr = match (canonical_extra_bytes_vlr, reference_parsed) {
        (Some(mut bytes), Some(_)) => {
            let per_file_parsed: Vec<ParsedExtraBytes> = results
                .iter()
                .filter_map(|r| r.extra_bytes_parsed.clone())
                .collect();
            if let Err(e) = merge_stats_into_canonical(&mut bytes, &per_file_parsed) {
                warn!("Could not merge Extra Bytes stats, using reference file's stats as-is: {e}");
            }
            Some(bytes)
        }
        (vlr, _) => vlr,
    };

    let point_format = input_to_copc_format(first_format);
    debug!(
        "Input point format: {first_format}, output COPC point format: {point_format}, \
         extra bytes per point: {reference_num_extras}"
    );

    Ok(ValidatedInputs {
        wkt_crs: canonical_wkt,
        extra_bytes_vlr,
        num_extra_bytes: reference_num_extras,
        point_format,
    })
}

/// Build an `Error::ExtraBytesMismatch` with a multi-line `detail`
/// string covering every structural difference between the reference
/// file and the offending file.
fn extras_mismatch_error(
    input_files: &[PathBuf],
    reference_idx: usize,
    reference_parsed: Option<&ParsedExtraBytes>,
    reference_num_extras: u16,
    other_idx: usize,
    other: &ScanResult,
) -> Error {
    let mut detail_lines: Vec<String> = Vec::new();

    if other.num_extra_bytes != reference_num_extras {
        detail_lines.push(format!(
            "  - point trailing-byte width differs: reference declares {} bytes, other declares {}",
            reference_num_extras, other.num_extra_bytes
        ));
    }

    match (reference_parsed, other.extra_bytes_parsed.as_ref()) {
        (Some(reference), Some(other_parsed)) => {
            for diff in diff_schemas(reference, other_parsed) {
                detail_lines.push(format!("  - {diff}"));
            }
        }
        (Some(_), None) => {
            detail_lines
                .push("  - reference declares an Extra Bytes VLR, other declares none".to_owned());
        }
        (None, Some(_)) => {
            detail_lines
                .push("  - other declares an Extra Bytes VLR, reference declares none".to_owned());
        }
        (None, None) => {}
    }

    let detail = if detail_lines.is_empty() {
        "  - (no structural differences detected; schema hashes diverged for an unknown reason)"
            .to_owned()
    } else {
        detail_lines.join("\n")
    };

    Error::ExtraBytesMismatch {
        file_a: input_files[reference_idx].clone(),
        file_b: input_files[other_idx].clone(),
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra_bytes::ParsedExtraBytes;
    use crate::octree::{Bounds, ScanResult};

    /// Build a minimal one-field Extra Bytes VLR payload with the given
    /// data type and per-axis min/max. Mirrors the helper in
    /// `extra_bytes::tests`, kept here so the validate tests can
    /// construct synthetic ScanResults without touching real LAS files.
    fn build_one_field_vlr(
        data_type: u8,
        name: &str,
        min0: f64,
        max0: f64,
    ) -> (Vec<u8>, ParsedExtraBytes) {
        let mut buf = vec![0u8; 192];
        buf[2] = data_type;
        buf[3] = 0b0000_0110; // min_bit + max_bit
        let nm = name.as_bytes();
        buf[4..4 + nm.len()].copy_from_slice(nm);
        // min at byte 64
        buf[64..72].copy_from_slice(&min0.to_le_bytes());
        // max at byte 88
        buf[88..96].copy_from_slice(&max0.to_le_bytes());
        let parsed = ParsedExtraBytes::parse(&buf).unwrap();
        (buf, parsed)
    }

    fn make_result(wkt_hash: Option<u64>, fmt: u8) -> ScanResult {
        ScanResult {
            bounds: Bounds::empty(),
            point_count: 100,
            scale_x: 0.001,
            scale_y: 0.001,
            scale_z: 0.001,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            wkt_crs_hash: wkt_hash,
            extra_bytes_parsed: None,
            extra_bytes_schema_hash: None,
            num_extra_bytes: 0,
            point_format_id: fmt,
        }
    }

    fn make_result_with_extras(
        wkt_hash: Option<u64>,
        parsed: ParsedExtraBytes,
        num_extra: u16,
        fmt: u8,
    ) -> ScanResult {
        let schema_hash = crate::extra_bytes::schema_hash(&parsed);
        ScanResult {
            bounds: Bounds::empty(),
            point_count: 100,
            scale_x: 0.001,
            scale_y: 0.001,
            scale_z: 0.001,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            wkt_crs_hash: wkt_hash,
            extra_bytes_parsed: Some(parsed),
            extra_bytes_schema_hash: Some(schema_hash),
            num_extra_bytes: num_extra,
            point_format_id: fmt,
        }
    }

    #[test]
    fn validate_single_file() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 3)];
        let v = validate(&files, &results, None, None, false).unwrap();
        assert_eq!(v.point_format, 7);
        assert!(v.wkt_crs.is_none());
        assert!(v.extra_bytes_vlr.is_none());
        assert_eq!(v.num_extra_bytes, 0);
    }

    #[test]
    fn validate_matching_files() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![make_result(Some(42), 8), make_result(Some(42), 8)];
        let v = validate(&files, &results, Some(b"WKT".to_vec()), None, false).unwrap();
        assert_eq!(v.point_format, 8);
        assert_eq!(v.wkt_crs.as_deref(), Some(&b"WKT"[..]));
    }

    #[test]
    fn validate_crs_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![make_result(Some(1), 7), make_result(Some(2), 7)];
        let err = validate(&files, &results, Some(b"WKT_A".to_vec()), None, false).unwrap_err();
        assert!(matches!(err, Error::CrsMismatch { .. }));
    }

    #[test]
    fn validate_format_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![make_result(None, 3), make_result(None, 7)];
        let err = validate(&files, &results, None, None, false).unwrap_err();
        assert!(matches!(err, Error::PointFormatMismatch { .. }));
    }

    #[test]
    fn validate_temporal_index_requires_gps_time() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 0)];
        let err = validate(&files, &results, None, None, true).unwrap_err();
        assert!(matches!(err, Error::NoGpsTime { .. }));
    }

    #[test]
    fn validate_temporal_index_with_gps_time() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 1)];
        let v = validate(&files, &results, None, None, true).unwrap();
        assert_eq!(v.point_format, 6);
    }

    #[test]
    fn validate_extras_matching_with_identical_stats() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(10, "zNorm", -1.0, 10.0);
        let (_, parsed_b) = build_one_field_vlr(10, "zNorm", -1.0, 10.0);
        let results = vec![
            make_result_with_extras(None, parsed_a, 8, 3),
            make_result_with_extras(None, parsed_b, 8, 3),
        ];
        let v = validate(&files, &results, None, Some(canonical), false).unwrap();
        assert_eq!(v.num_extra_bytes, 8);
        let merged = ParsedExtraBytes::parse(v.extra_bytes_vlr.as_ref().unwrap()).unwrap();
        assert_eq!(merged.stats[0].min[0], -1.0);
        assert_eq!(merged.stats[0].max[0], 10.0);
    }

    /// The original motivating bug: two files declare the *same logical
    /// schema* but have different per-file stats (different min/max
    /// because the data ranges differ). v0.9.11 byte-hashed the VLRs and
    /// erroneously rejected this. The new schema-hash check must accept
    /// the inputs and merge stats honestly into the output.
    #[test]
    fn validate_extras_same_schema_different_stats_is_accepted() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(10, "zNorm", -4.5, 10.0);
        let (_, parsed_b) = build_one_field_vlr(10, "zNorm", -2.0, 24.7);
        let results = vec![
            make_result_with_extras(None, parsed_a, 8, 3),
            make_result_with_extras(None, parsed_b, 8, 3),
        ];
        let v = validate(&files, &results, None, Some(canonical), false).unwrap();
        let merged = ParsedExtraBytes::parse(v.extra_bytes_vlr.as_ref().unwrap()).unwrap();
        // Output stats must be the union: min from a, max from b.
        assert_eq!(merged.stats[0].min[0], -4.5);
        assert_eq!(merged.stats[0].max[0], 24.7);
    }

    #[test]
    fn validate_extras_data_type_mismatch_lists_field() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(3, "treeId", 0.0, 0.0); // u16
        let (_, parsed_b) = build_one_field_vlr(5, "treeId", 0.0, 0.0); // u32
        let results = vec![
            make_result_with_extras(None, parsed_a, 2, 3),
            make_result_with_extras(None, parsed_b, 4, 3),
        ];
        let err = validate(&files, &results, None, Some(canonical), false).unwrap_err();
        let Error::ExtraBytesMismatch {
            file_a,
            file_b,
            detail,
        } = err
        else {
            panic!("expected ExtraBytesMismatch");
        };
        assert_eq!(file_a, PathBuf::from("a.laz"));
        assert_eq!(file_b, PathBuf::from("b.laz"));
        // Trailing-byte width mismatch (u16 -> 2 bytes, u32 -> 4 bytes)
        // and per-field data_type difference must both be reported.
        assert!(
            detail.contains("trailing-byte width"),
            "expected width line, got:\n{detail}"
        );
        assert!(
            detail.contains("data_type") && detail.contains("treeId"),
            "expected per-field data_type diff for treeId, got:\n{detail}"
        );
        assert!(detail.contains("u16"));
        assert!(detail.contains("u32"));
    }

    #[test]
    fn validate_extras_one_file_lacks_them() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(10, "zNorm", 0.0, 0.0);
        let results = vec![
            make_result_with_extras(None, parsed_a, 8, 3),
            make_result(None, 3), // no extras
        ];
        let err = validate(&files, &results, None, Some(canonical), false).unwrap_err();
        let Error::ExtraBytesMismatch { detail, .. } = err else {
            panic!("expected ExtraBytesMismatch");
        };
        assert!(
            detail.contains("declares an Extra Bytes VLR") && detail.contains("declares none"),
            "expected presence-mismatch line, got:\n{detail}"
        );
    }

    /// All differences must be listed in a single error, not just the
    /// first one — so the user can fix the whole input set in one pass
    /// rather than re-running once per discovered problem.
    #[test]
    fn validate_extras_lists_all_differences() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(3, "treeId", 0.0, 0.0);
        let (mut other_bytes, _) = build_one_field_vlr(5, "treeId", 0.0, 0.0);
        // Force a description mismatch on top of the data_type mismatch.
        let desc = b"different desc";
        other_bytes[160..160 + desc.len()].copy_from_slice(desc);
        let parsed_b = ParsedExtraBytes::parse(&other_bytes).unwrap();
        let results = vec![
            make_result_with_extras(None, parsed_a, 2, 3),
            make_result_with_extras(None, parsed_b, 4, 3),
        ];
        let err = validate(&files, &results, None, Some(canonical), false).unwrap_err();
        let Error::ExtraBytesMismatch { detail, .. } = err else {
            panic!("expected ExtraBytesMismatch");
        };
        assert!(
            detail.contains("data_type"),
            "missing data_type diff in: {detail}"
        );
        assert!(
            detail.contains("description"),
            "missing description diff in: {detail}"
        );
    }
}
