use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deep_array::Array;

fn criterion_benchmark(c: &mut Criterion) {
    println!("benchmarking");
    let original: Array<usize, 3> = Array::fill(1, [10, 10, 10]);
    println!("benchmarking");
    c.bench_function("Clone usize 10x10x10", |b| {
        b.iter(|| black_box(original.clone()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

// use criterion::{black_box, criterion_group, criterion_main, Criterion};
//
// fn fibonacci(n: u64) -> u64 {
//     match n {
//         0 => 1,
//         1 => 1,
//         n => fibonacci(n-1) + fibonacci(n-2),
//     }
// }
//
// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
// }
//
// criterion_group!(benches, criterion_benchmark);
// criterion_main!(benches);
