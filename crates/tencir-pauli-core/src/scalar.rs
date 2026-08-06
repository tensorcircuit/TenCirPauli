use std::hash::{Hash, Hasher};

pub use num_complex::Complex64;

pub(crate) fn is_exact_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

/// Hash one binary64 value with both signed zeros represented identically.
pub fn hash_f64<H: Hasher>(value: f64, state: &mut H) {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    bits.hash(state);
}

/// Hash a complex coefficient using the canonical binary64 zero rule.
pub fn hash_complex<H: Hasher>(value: Complex64, state: &mut H) {
    hash_f64(value.re, state);
    hash_f64(value.im, state);
}
