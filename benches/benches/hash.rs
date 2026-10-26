use crate::prelude::*;
use core::hash::BuildHasher;
use foldhash::fast::FixedState;

pub fn group(criterion: &mut Criterion) {
    let state = FixedState::with_seed(0x9e37_79b9_7f4a_7c15);
    for &size in &SIZE_NIBBLES {
        bench_unop(criterion, "hash", size, |n| state.hash_one(n));
    }
}
