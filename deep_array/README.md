# Deep Array
This is a custom, high-performance array library targeted at my [deep learning library](https://www.github.com/Wuelle/deep_thought).
Because of this, its going to specialize in Level 3 BLAS Operations (`gemm`).

I am currently writing a blog about the development of this crate, you can find it [here](https://wuelle.github.io/array_docs).

## Debugging
Enable the `debug_allocator` feature during debugging to log all memory allocations/deallocations.
They are logged with the `trace` level, so depending on your logger you might have to enable that.
All tests use the [`env-logger`](https://github.com/env-logger-rs/env_logger/) crate, so to see logging output you will have to do something like this:
```
RUST_LOG=trace cargo t <testname> --features debug_allocator
```
