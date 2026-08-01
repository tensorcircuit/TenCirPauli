pub use num_complex::Complex64;

pub(crate) fn is_exact_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}
