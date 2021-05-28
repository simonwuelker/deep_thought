use ndarray::Array1;

/// replace negative values with zero
pub fn relu(x: Array1<f32>) -> Array1<f32> {
    //  std::cmp::max(0.0, x);
    unimplemented!()
}

/// squash every input into a range between -1 and 1
pub fn sigmoid(x: f32) -> f32 {
    unimplemented!();
}
