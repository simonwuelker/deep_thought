use ndarray::Array1;
// TODO: replace this import
use ndarray_rand::rand_distr::num_traits::Pow;

/// replace negative values with zero
pub fn relu(x: &mut f32) {
    if *x < 0.0 { *x = 0.0 };
}

/// squash every input into a range between -1 and 1
pub fn sigmoid(x: &mut f32) {
    *x = 1.0 / (1.0 + (2.71_f32.pow(-1.0 * *x)))
}
