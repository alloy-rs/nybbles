use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use nybbles::Nibbles;
use proptest::{prelude::*, strategy::ValueTree};
use std::{hint::black_box, time::Duration};

/// Benchmarks the nibble unpacking.
pub fn nibbles_benchmark(c: &mut Criterion) {
    let lengths = [8u64, 16, 32];

    {
        let mut g = group(c, "unpack");
        for len in lengths {
            g.throughput(criterion::Throughput::Bytes(len));

            let id = criterion::BenchmarkId::new("naive", len);
            g.bench_function(id, |b| {
                let bytes = &get_bytes(len as usize)[..];
                b.iter(|| unpack_naive(black_box(bytes)))
            });

            let id = criterion::BenchmarkId::new("nybbles", len);
            g.bench_function(id, |b| {
                let bytes = &get_bytes(len as usize)[..];
                b.iter(|| Nibbles::unpack(black_box(bytes)))
            });
        }
    }

    {
        let mut g = group(c, "pack");
        for len in lengths {
            g.throughput(criterion::Throughput::Bytes(len));

            let id = criterion::BenchmarkId::new("naive", len);
            g.bench_function(id, |b| {
                let bytes = &get_nibbles(len as usize).to_vec();
                b.iter(|| pack_naive(black_box(bytes)))
            });

            let id = criterion::BenchmarkId::new("nybbles", len);
            g.bench_function(id, |b| {
                let bytes = &get_nibbles(len as usize);
                b.iter(|| black_box(bytes).pack())
            });
        }
    }
}

fn group<'c>(c: &'c mut Criterion, name: &str) -> BenchmarkGroup<'c, WallTime> {
    let mut g = c.benchmark_group(name);
    g.warm_up_time(Duration::from_secs(1));
    g.noise_threshold(0.02);
    g
}

fn get_nibbles(len: usize) -> Nibbles {
    proptest::arbitrary::any_with::<Nibbles>(len.into())
        .new_tree(&mut Default::default())
        .unwrap()
        .current()
}

fn get_bytes(len: usize) -> Vec<u8> {
    proptest::collection::vec(proptest::arbitrary::any::<u8>(), len)
        .new_tree(&mut Default::default())
        .unwrap()
        .current()
}

fn unpack_naive(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|byte| [byte >> 4, byte & 0x0f]).collect()
}

fn pack_naive(bytes: &[u8]) -> Vec<u8> {
    let chunks = bytes.chunks_exact(2);
    let rem = chunks.remainder();
    chunks.map(|chunk| (chunk[0] << 4) | chunk[1]).chain(rem.iter().copied()).collect()
}

criterion_group!(benches, nibbles_benchmark);
criterion_main!(benches);

#[test]
fn naive_equivalency() {
    for len in [0, 1, 2, 3, 4, 15, 16, 17, 31, 32, 33] {
        let bytes = get_bytes(len);
        let nibbles = Nibbles::unpack(&bytes);
        assert_eq!(unpack_naive(&bytes), nibbles.to_vec());
        assert_eq!(pack_naive(&nibbles.to_vec())[..], nibbles.pack()[..]);
    }
}
