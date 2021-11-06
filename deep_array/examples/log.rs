use deep_array::Array;

fn main() {
    #[cfg(feature = "log")]
    env_logger::init();

    let _a: Array<usize, 1> = Array::fill(1, [2]);
}
