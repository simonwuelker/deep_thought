use log::{trace, warn};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt;

/// This allocator will log all memory allocations/deallocations. It is only meant
/// for debugging and should never be used in production.
/// It wraps around the Systems default allocator (`malloc` on UNIX, `HeapAlloc` on windows).
struct DebugAllocator;

/// Execute a closure without logging on allocations.
/// This prevents infinite loops (allocation -> log -> allocation -> log etc)
pub fn run_guarded<F>(f: F)
where
    F: FnOnce(),
{
    thread_local! {
        static GUARD: Cell<bool> = Cell::new(false);
    }

    GUARD.with(|guard| {
        if !guard.replace(true) {
            f();
            guard.set(false)
        }
    })
}

unsafe impl GlobalAlloc for DebugAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if ptr.is_null() {
            run_guarded(|| {
                warn!(
                    "{:<15} {}",
                    "failed alloc",
                    Operation(ptr, layout.size(), layout.align())
                )
            });
        } else {
            run_guarded(|| {
                trace!(
                    "{:<15} {}",
                    "alloc",
                    Operation(ptr, layout.size(), layout.align())
                )
            });
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        run_guarded(|| {
            trace!(
                "{:<15} {}",
                "dealloc",
                Operation(ptr, layout.size(), layout.align())
            )
        });
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        run_guarded(|| {
            trace!(
                "{:<15} {}",
                "alloc_zeroed",
                Operation(ptr, layout.size(), layout.align())
            );
        });
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        run_guarded(|| {
            trace!(
                "{:<15} {} to {}",
                "realloc",
                Operation(ptr, layout.size(), layout.align()),
                Operation(new_ptr, new_size, layout.align())
            );
        });
        new_ptr
    }
}

struct Operation(*mut u8, usize, usize);

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[address={:p}, size={:#04x}, align={:#04x}]",
            self.0, self.1, self.2
        )
    }
}

#[global_allocator]
static GLOBAL: DebugAllocator = DebugAllocator;
