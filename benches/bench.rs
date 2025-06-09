use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use nybbles::Nibbles;
use proptest::{prelude::*, strategy::ValueTree};
use std::{hint::black_box, time::Duration};

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

pub fn unpack(c: &mut Criterion) {
    let lengths = [8, 16, 32];

    let mut g = group(c, "unpack");
    for len in lengths {
        g.throughput(criterion::Throughput::Bytes(len as u64));

        let id = criterion::BenchmarkId::new("naive", len);
        g.bench_function(id, |b| {
            let bytes = &get_bytes(len)[..];
            b.iter(|| unpack_naive(black_box(bytes)))
        });

        let id = criterion::BenchmarkId::new("nybbles", len);
        g.bench_function(id, |b| {
            let bytes = &get_bytes(len)[..];
            b.iter(|| Nibbles::unpack(black_box(bytes)))
        });
    }
}

pub fn pack(c: &mut Criterion) {
    let lengths = [8, 16, 32];

    let mut g = group(c, "pack");
    for len in lengths {
        g.throughput(criterion::Throughput::Bytes(len as u64));

        let id = criterion::BenchmarkId::new("naive", len);
        g.bench_function(id, |b| {
            let bytes = &get_nibbles(len).to_vec();
            b.iter(|| pack_naive(black_box(bytes)))
        });

        let id = criterion::BenchmarkId::new("nybbles", len);
        g.bench_function(id, |b| {
            let bytes = &get_nibbles(len);
            b.iter(|| black_box(bytes).pack())
        });
    }
}

pub fn ord(c: &mut Criterion) {
    let lengths = [8, 16, 32, 64, 128];

    let mut g = group(c, "ord");
    for len in lengths {
        g.throughput(criterion::Throughput::Elements(1));

        let id = criterion::BenchmarkId::new("cmp_equal", len);
        g.bench_function(id, |b| {
            let nibbles1 = get_nibbles(len);
            let nibbles2 = nibbles1;
            b.iter(|| black_box(&nibbles1).cmp(black_box(&nibbles2)))
        });

        let id = criterion::BenchmarkId::new("cmp_different_same_length", len);
        g.bench_function(id, |b| {
            let nibbles1 = get_nibbles(len);
            let mut nibbles2 = get_nibbles(len);
            // Ensure they're different
            if nibbles1 == nibbles2 {
                nibbles2.push(0);
                nibbles2.pop();
            }
            b.iter(|| black_box(&nibbles1).cmp(black_box(&nibbles2)))
        });

        let id = criterion::BenchmarkId::new("cmp_different_lengths", len);
        g.bench_function(id, |b| {
            let nibbles1 = get_nibbles(len);
            let nibbles2 = get_nibbles(len / 2);
            b.iter(|| black_box(&nibbles1).cmp(black_box(&nibbles2)))
        });

        let id = criterion::BenchmarkId::new("cmp_prefix", len);
        g.bench_function(id, |b| {
            let mut nibbles1 = get_nibbles(len / 2);
            let nibbles2 = nibbles1;
            // Make nibbles1 longer with the same prefix
            for _ in 0..len / 2 {
                nibbles1.push(1);
            }
            b.iter(|| black_box(&nibbles1).cmp(black_box(&nibbles2)))
        });
    }
}

criterion_group!(benches, unpack, pack, ord);
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
