/// Validate consistency of scanned input files before building the octree.
use crate::Error;
use crate::octree::{ScanResult, input_to_copc_format};
use std::path::PathBuf;
use tracing::debug;

/// Validated output: consistent properties across all input files.
#[derive(Debug)]
pub struct ValidatedInputs {
    /// WKT CRS string from input files (if present). Single canonical copy
    /// from the first scanned file — per-file WKT bytes are discarded
    /// after hashing during scan to keep memory O(1) in file count.
    pub wkt_crs: Option<Vec<u8>>,
    /// LAS Extra Bytes VLR payload (if present). Single canonical copy
    /// from the first scanned file with the same memory rationale as
    /// `wkt_crs`. `None` when no input declares extra bytes.
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
    let extra_bytes_hash = first.extra_bytes_vlr_hash;
    let num_extra_bytes = first.num_extra_bytes;
    let first_format = first.point_format_id;

    for (i, r) in results.iter().enumerate().skip(1) {
        if r.wkt_crs_hash != wkt_hash {
            return Err(Error::CrsMismatch {
                file_a: input_files[0].clone(),
                file_b: input_files[i].clone(),
            });
        }
        if r.extra_bytes_vlr_hash != extra_bytes_hash || r.num_extra_bytes != num_extra_bytes {
            return Err(Error::ExtraBytesMismatch {
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

    if temporal_index && !format_has_gps_time(first_format) {
        return Err(Error::NoGpsTime {
            format: first_format,
        });
    }

    let point_format = input_to_copc_format(first_format);
    debug!(
        "Input point format: {first_format}, output COPC point format: {point_format}, \
         extra bytes per point: {num_extra_bytes}"
    );

    Ok(ValidatedInputs {
        wkt_crs: canonical_wkt,
        extra_bytes_vlr: canonical_extra_bytes_vlr,
        num_extra_bytes,
        point_format,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{Bounds, ScanResult};

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
            extra_bytes_vlr_hash: None,
            num_extra_bytes: 0,
            point_format_id: fmt,
        }
    }

    fn make_result_with_extras(
        wkt_hash: Option<u64>,
        extra_hash: Option<u64>,
        num_extra: u16,
        fmt: u8,
    ) -> ScanResult {
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
            extra_bytes_vlr_hash: extra_hash,
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
    fn validate_extras_matching() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result_with_extras(None, Some(7), 12, 3),
            make_result_with_extras(None, Some(7), 12, 3),
        ];
        let v = validate(&files, &results, None, Some(b"VLR".to_vec()), false).unwrap();
        assert_eq!(v.num_extra_bytes, 12);
        assert_eq!(v.extra_bytes_vlr.as_deref(), Some(&b"VLR"[..]));
    }

    #[test]
    fn validate_extras_vlr_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result_with_extras(None, Some(7), 12, 3),
            make_result_with_extras(None, Some(8), 12, 3),
        ];
        let err = validate(&files, &results, None, Some(b"VLR".to_vec()), false).unwrap_err();
        assert!(matches!(err, Error::ExtraBytesMismatch { .. }));
    }

    #[test]
    fn validate_extras_count_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result_with_extras(None, Some(7), 12, 3),
            make_result_with_extras(None, Some(7), 8, 3),
        ];
        let err = validate(&files, &results, None, Some(b"VLR".to_vec()), false).unwrap_err();
        assert!(matches!(err, Error::ExtraBytesMismatch { .. }));
    }

    #[test]
    fn validate_extras_one_file_lacks_them() {
        // One file declares extras, another doesn't — schema mismatch.
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result_with_extras(None, Some(7), 12, 3),
            make_result_with_extras(None, None, 0, 3),
        ];
        let err = validate(&files, &results, None, Some(b"VLR".to_vec()), false).unwrap_err();
        assert!(matches!(err, Error::ExtraBytesMismatch { .. }));
    }
}
