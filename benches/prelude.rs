#![allow(unexpected_cfgs)]
//! Benchmark prelude providing CodSpeed-compatible helpers.
//!
//! The challenge with CodSpeed is that it only uses 1 input regardless of the function used,
//! so traditional `iter_batched` with random input generation doesn't provide variability.
//! This module provides wrapper functions that run multiple iterations with different inputs
//! within a single benchmark call, making them effective both locally and on CodSpeed.

use std::cell::RefCell;

use criterion::BatchSize;
use nybbles::Nibbles;
use proptest::{prelude::Strategy, strategy::ValueTree, test_runner::TestRunner};

#[allow(dead_code)]
const CODSPEED_BATCH_SIZE: usize = 100;

pub fn arbitrary_nibbles(size: usize) -> impl Strategy<Value = Nibbles> + Clone {
    proptest::collection::vec(0u8..16, size).prop_map(Nibbles::from_nibbles)
}

pub fn arbitrary_non_empty_nibbles(size: usize) -> impl Strategy<Value = Nibbles> + Clone {
    proptest::collection::vec(0u8..16, 1..size).prop_map(Nibbles::from_nibbles)
}

pub fn arbitrary_raw_nibbles(size: usize) -> impl Strategy<Value = Vec<u8>> + Clone {
    proptest::collection::vec(0u8..16, size)
}

pub fn arbitrary_bytes(size: usize) -> impl Strategy<Value = Vec<u8>> + Clone {
    proptest::collection::vec(proptest::arbitrary::any::<u8>(), size)
}

pub fn bench_arbitrary_with<T: Strategy, U>(
    criterion: &mut criterion::Criterion,
    name: impl AsRef<str>,
    input: T,
    f: impl FnMut(&T::Value) -> U,
) {
    let name = name.as_ref();
    let runner = std::cell::RefCell::new(TestRunner::deterministic());
    let mut setup = mk_setup(&input, &runner);
    let mut f = manual_batch(mk_setup(&input, &runner), f);
    criterion.bench_function(name, move |bencher| {
        bencher.iter_batched(&mut setup, &mut f, BatchSize::SmallInput);
    });
}

fn mk_setup<'a, T: Strategy>(
    input: &'a T,
    runner: &'a RefCell<TestRunner>,
) -> impl FnMut() -> T::Value + 'a {
    move || input.new_tree(&mut runner.borrow_mut()).unwrap().current()
}

/// Codspeed does not batch inputs even if `iter_batched` is used, so we have to
/// do it ourselves for operations that would otherwise be too fast to be
/// measured accurately.
#[cfg(codspeed)]
#[inline]
fn manual_batch<T, U>(mut setup: impl FnMut() -> T, mut f: impl FnMut(&T) -> U) -> impl FnMut(T) {
    let inputs =
        criterion::black_box((0..CODSPEED_BATCH_SIZE).map(|_| setup()).collect::<Vec<_>>());
    let mut out = criterion::black_box(Box::new_uninit_slice(CODSPEED_BATCH_SIZE));
    move |_| {
        for i in 0..criterion::black_box(CODSPEED_BATCH_SIZE) {
            let input = unsafe { inputs.get_unchecked(i) };
            let output = unsafe { out.get_unchecked_mut(i) };
            output.write(f(input));
        }
    }
}

#[cfg(not(codspeed))]
fn manual_batch<T, U>(_setup: impl FnMut() -> T, mut f: impl FnMut(&T) -> U) -> impl FnMut(T) {
    move |input| {
        f(&input);
    }
}
