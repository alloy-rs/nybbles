use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nybbles::Nibbles;
use proptest::{prelude::*, strategy::ValueTree};
use std::{hint::black_box, time::Duration};

const SIZE_NIBBLES: [usize; 4] = [8, 16, 32, 64];
const SIZE_BYTES: [usize; 4] = [4, 8, 16, 32];

pub fn bench_from_nibbles(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_nibbles");

    for size in SIZE_NIBBLES {
        let nibbles_data = generate_nibbles(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &nibbles_data, |b, data| {
            b.iter(|| Nibbles::from_nibbles(black_box(data)))
        });
    }

    group.finish();
}

pub fn bench_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack");

    for &size in &SIZE_BYTES {
        let bytes = generate_bytes(size);
        let nibbles = Nibbles::unpack(&bytes);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("pack", size), &nibbles, |b, data| {
            b.iter(|| black_box(data).pack())
        });

        let output = vec![0u8; nibbles.len().div_ceil(2)];
        group.bench_with_input(
            BenchmarkId::new("pack_to", size),
            &(nibbles, output),
            |b, (data, buf)| {
                b.iter_batched(
                    || buf.clone(),
                    |mut buffer| {
                        black_box(data).pack_to(black_box(&mut buffer));
                        buffer
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

pub fn bench_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpack");

    for &size in &SIZE_BYTES {
        let bytes = generate_bytes(size);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &bytes, |b, data| {
            b.iter(|| Nibbles::unpack(black_box(data)))
        });
    }

    group.finish();
}

pub fn bench_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("push");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        let nibbles = generate_nibbles(size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &nibbles, |b, nibbles| {
            b.iter(|| {
                let mut nib = Nibbles::new();
                for nibble in nibbles {
                    nib.push(black_box(*nibble));
                }
                nib
            })
        });
    }

    group.finish();
}

pub fn bench_push_unchecked(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_unchecked");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        let nibbles = generate_nibbles(size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &nibbles, |b, nibbles| {
            b.iter(|| {
                let mut nib = Nibbles::new();
                for nibble in nibbles {
                    nib.push_unchecked(black_box(*nibble));
                }
                nib
            })
        });
    }

    group.finish();
}

pub fn bench_push_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_comparison");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        let nibbles = generate_nibbles(size);

        group.bench_with_input(BenchmarkId::new("push", size), &nibbles, |b, nibbles| {
            b.iter(|| {
                let mut nib = Nibbles::new();
                for nibble in nibbles {
                    nib.push(black_box(*nibble));
                }
                nib
            })
        });

        group.bench_with_input(BenchmarkId::new("push_unchecked", size), &nibbles, |b, nibbles| {
            b.iter(|| {
                let mut nib = Nibbles::new();
                for nibble in nibbles {
                    nib.push_unchecked(black_box(*nibble));
                }
                nib
            })
        });
    }

    group.finish();
}

pub fn bench_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice");

    for size in SIZE_NIBBLES {
        let nibbles = Nibbles::from_nibbles(generate_nibbles(size));

        group.bench_with_input(BenchmarkId::new("from_start", size), &nibbles, |b, data| {
            let end = data.len() / 2;
            b.iter(|| black_box(data).slice(black_box(0..end)))
        });
        group.bench_with_input(BenchmarkId::new("middle", size), &nibbles, |b, data| {
            let start = data.len() / 4;
            let end = data.len() / 2;
            b.iter(|| black_box(data).slice(black_box(start..end)))
        });
        group.bench_with_input(BenchmarkId::new("to_end", size), &nibbles, |b, data| {
            let start = data.len() / 2;
            b.iter(|| black_box(data).slice(black_box(start..)))
        });
    }

    group.finish();
}

pub fn bench_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("join");

    for size in SIZE_NIBBLES {
        let nibbles = Nibbles::from_nibbles(generate_nibbles(size));
        let other_nibbles = Nibbles::from_nibbles(generate_nibbles(size / 2));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(nibbles, other_nibbles),
            |b, (data, other)| b.iter(|| black_box(data).join(black_box(other))),
        );
    }

    group.finish();
}

pub fn bench_extend(c: &mut Criterion) {
    let mut group = c.benchmark_group("extend");

    for &size in &SIZE_NIBBLES[..SIZE_NIBBLES.len() - 1] {
        let nibbles = Nibbles::from_nibbles(generate_nibbles(size));
        let other_nibbles = Nibbles::from_nibbles(generate_nibbles(size));

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(nibbles, other_nibbles),
            |b, (data, other)| {
                b.iter_batched(
                    || *data,
                    |mut nib| {
                        nib.extend(black_box(other));
                        nib
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

pub fn bench_set_at(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_at");

    for size in SIZE_NIBBLES {
        let nibbles = Nibbles::from_nibbles(generate_nibbles(size));

        group.bench_with_input(BenchmarkId::from_parameter(size), &nibbles, |b, data| {
            b.iter_batched(
                || *data,
                |mut nib| {
                    for i in 0..nib.len() {
                        nib.set_at(black_box(i), black_box((i % 16) as u8));
                    }
                    nib
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_get_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_byte");

    for size in SIZE_NIBBLES {
        let nibbles = Nibbles::from_nibbles(generate_nibbles(size));

        group.bench_with_input(BenchmarkId::new("get_byte", size), &nibbles, |b, data| {
            b.iter(|| {
                let mut sum = 0u64;
                for i in 0..data.len().saturating_sub(1) {
                    if let Some(byte) = data.get_byte(black_box(i)) {
                        sum = sum.wrapping_add(byte as u64);
                    }
                }
                sum
            })
        });

        group.bench_with_input(
            BenchmarkId::new("get_byte_unchecked", size),
            &nibbles,
            |b, data| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for i in 0..data.len().saturating_sub(1) {
                        let byte = data.get_byte_unchecked(black_box(i));
                        sum = sum.wrapping_add(byte as u64);
                    }
                    sum
                })
            },
        );
    }

    group.finish();
}

pub fn bench_common_prefix_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("common_prefix_length");

    for size in SIZE_NIBBLES {
        let nibbles_a = Nibbles::from_nibbles(generate_nibbles(size));
        let nibbles_b = nibbles_a.slice(..nibbles_a.len() - 1);

        group.throughput(Throughput::Elements(nibbles_b.len() as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(nibbles_a, nibbles_b),
            |b, (data, other)| b.iter(|| data.common_prefix_length(black_box(other))),
        );
    }

    group.finish();
}

pub fn bench_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp");

    for size in SIZE_NIBBLES {
        let nibbles_a = Nibbles::from_nibbles(generate_nibbles(size));
        let nibbles_b = Nibbles::from_nibbles(generate_nibbles(size));

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(nibbles_a, nibbles_b),
            |b, (data, other)| b.iter(|| black_box(data).cmp(black_box(other))),
        );
    }

    group.finish();
}

pub fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone");

    for size in SIZE_NIBBLES {
        let nibbles = Nibbles::from_nibbles(generate_nibbles(size));

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &nibbles, |b, data| {
            b.iter(|| black_box(*data))
        });
    }

    group.finish();
}

pub fn bench_increment(c: &mut Criterion) {
    let mut group = c.benchmark_group("increment");

    for size in SIZE_NIBBLES {
        let nibbles = Nibbles::from_nibbles(generate_nibbles(size));

        group.bench_with_input(BenchmarkId::from_parameter(size), &nibbles, |b, data| {
            b.iter(|| black_box(data).increment())
        });
    }

    group.finish();
}

pub fn nibbles_benchmark(c: &mut Criterion) {
    {
        let mut g = c.benchmark_group("unpack");
        for size in SIZE_BYTES {
            g.throughput(criterion::Throughput::Bytes(size as u64));

            let id = criterion::BenchmarkId::new("naive", size);
            g.bench_function(id, |b| {
                let bytes = &generate_bytes(size)[..];
                b.iter(|| unpack_naive(black_box(bytes)))
            });

            let id = criterion::BenchmarkId::new("nybbles", size);
            g.bench_function(id, |b| {
                let bytes = &generate_bytes(size)[..];
                b.iter(|| Nibbles::unpack(black_box(bytes)))
            });
        }
    }

    {
        let mut g = c.benchmark_group("pack");
        for size in SIZE_NIBBLES {
            g.throughput(criterion::Throughput::Elements(size as u64));

            let id = criterion::BenchmarkId::new("naive", size);
            g.bench_function(id, |b| {
                let bytes = &get_nibbles(size).to_vec();
                b.iter(|| pack_naive(black_box(bytes)))
            });

            let id = criterion::BenchmarkId::new("nybbles", size);
            g.bench_function(id, |b| {
                let bytes = &get_nibbles(size);
                b.iter(|| black_box(bytes).pack())
            });
        }
    }
}

fn generate_bytes(len: usize) -> Vec<u8> {
    proptest::collection::vec(proptest::arbitrary::any::<u8>(), len)
        .new_tree(&mut Default::default())
        .unwrap()
        .current()
}

fn generate_nibbles(len: usize) -> Vec<u8> {
    proptest::collection::vec(0u8..16, len).new_tree(&mut Default::default()).unwrap().current()
}

fn get_nibbles(len: usize) -> Nibbles {
    Nibbles::from_nibbles(generate_nibbles(len))
}

fn unpack_naive(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|byte| [byte >> 4, byte & 0x0f]).collect()
}

fn pack_naive(bytes: &[u8]) -> Vec<u8> {
    let chunks = bytes.chunks_exact(2);
    let rem = chunks.remainder();
    chunks.map(|chunk| (chunk[0] << 4) | chunk[1]).chain(rem.iter().copied()).collect()
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500));
    targets = bench_from_nibbles, bench_pack, bench_unpack, bench_push, bench_push_unchecked, bench_push_comparison, bench_slice,
              bench_join, bench_extend, bench_set_at, bench_get_byte, bench_common_prefix_length,
              bench_cmp, bench_clone, bench_increment, nibbles_benchmark
);
criterion_main!(benches);

#[test]
fn naive_equivalency() {
    for len in [0, 1, 2, 3, 4, 15, 16, 17, 31, 32, 33] {
        let bytes = generate_bytes(len);
        let nibbles = Nibbles::unpack(&bytes);
        assert_eq!(unpack_naive(&bytes), nibbles.to_vec());
        assert_eq!(pack_naive(&nibbles.to_vec())[..], nibbles.pack()[..]);
    }
}
