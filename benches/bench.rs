use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nybbles::Nibbles;
use rand::{thread_rng, Rng};
use std::{hint::black_box, time::Duration};

const SIZE_NIBBLES: [usize; 4] = [8, 16, 32, 64];
const SIZE_BYTES: [usize; 4] = [4, 8, 16, 32];

pub fn bench_from_nibbles(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_nibbles");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || generate_nibbles_random(size),
                |data| Nibbles::from_nibbles(black_box(data)),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack");

    for &size in &SIZE_BYTES {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function(BenchmarkId::new("pack", size), |b| {
            b.iter_batched(
                || {
                    let bytes = generate_bytes_random(size);
                    Nibbles::unpack(&bytes)
                },
                |data| black_box(data).pack(),
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_function(BenchmarkId::new("pack_to", size), |b| {
            b.iter_batched(
                || {
                    let bytes = generate_bytes_random(size);
                    let nibbles = Nibbles::unpack(&bytes);
                    let output = vec![0u8; nibbles.len().div_ceil(2)];
                    (nibbles, output)
                },
                |(data, mut buffer)| {
                    black_box(&data).pack_to(black_box(&mut buffer));
                    buffer
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpack");

    for &size in &SIZE_BYTES {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || generate_bytes_random(size),
                |data| Nibbles::unpack(black_box(&data)),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("push");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || generate_nibbles_random(size),
                |nibbles| {
                    let mut nib = Nibbles::new();
                    for nibble in nibbles {
                        nib.push(black_box(nibble));
                    }
                    nib
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice");

    for size in SIZE_NIBBLES {
        group.bench_function(BenchmarkId::new("from_start", size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| {
                    let end = data.len() / 2;
                    black_box(&data).slice(black_box(0..end))
                },
                criterion::BatchSize::SmallInput,
            )
        });
        group.bench_function(BenchmarkId::new("middle", size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| {
                    let start = data.len() / 4;
                    let end = data.len() / 2;
                    black_box(&data).slice(black_box(start..end))
                },
                criterion::BatchSize::SmallInput,
            )
        });
        group.bench_function(BenchmarkId::new("to_end", size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| {
                    let start = data.len() / 2;
                    black_box(&data).slice(black_box(start..))
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("join");

    for size in SIZE_NIBBLES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || {
                    let nibbles = Nibbles::from_nibbles(generate_nibbles_random(size));
                    let other_nibbles = Nibbles::from_nibbles(generate_nibbles_random(size / 2));
                    (nibbles, other_nibbles)
                },
                |(a, b_nib)| black_box(&a).join(black_box(&b_nib)),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_extend(c: &mut Criterion) {
    let mut group = c.benchmark_group("extend");

    for &size in &SIZE_NIBBLES[..SIZE_NIBBLES.len() - 1] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || {
                    let nibbles = Nibbles::from_nibbles(generate_nibbles_random(size));
                    let other_nibbles = Nibbles::from_nibbles(generate_nibbles_random(size));
                    (nibbles, other_nibbles)
                },
                |(mut nib, b_nib)| {
                    nib.extend_from_slice(black_box(&b_nib));
                    nib
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_set_at(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_at");

    for size in SIZE_NIBBLES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
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
        group.bench_function(BenchmarkId::new("get_byte", size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| {
                    let mut sum = 0u64;
                    for i in 0..data.len().saturating_sub(1) {
                        if let Some(byte) = data.get_byte(black_box(i)) {
                            sum = sum.wrapping_add(byte as u64);
                        }
                    }
                    sum
                },
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_function(BenchmarkId::new("get_byte_unchecked", size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| {
                    let mut sum = 0u64;
                    for i in 0..data.len().saturating_sub(1) {
                        unsafe {
                            let byte = data.get_byte_unchecked(black_box(i));
                            sum = sum.wrapping_add(byte as u64);
                        }
                    }
                    sum
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_common_prefix_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("common_prefix_length");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements((size - 1) as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || {
                    let nibbles_a = Nibbles::from_nibbles(generate_nibbles_random(size));
                    let nibbles_b = nibbles_a.slice(..nibbles_a.len() - 1);
                    (nibbles_a, nibbles_b)
                },
                |(data, other)| data.common_prefix_length(black_box(&other)),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmp");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || {
                    let nibbles_a = Nibbles::from_nibbles(generate_nibbles_random(size));
                    let nibbles_b = Nibbles::from_nibbles(generate_nibbles_random(size));
                    (nibbles_a, nibbles_b)
                },
                |(data, other)| black_box(&data).cmp(black_box(&other)),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| black_box(&data).clone(),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_increment(c: &mut Criterion) {
    let mut group = c.benchmark_group("increment");

    for size in SIZE_NIBBLES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| black_box(&data).increment(),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |mut nib| {
                    for _ in 0..nib.len() {
                        black_box(nib.pop());
                    }
                    nib
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_from_vec_unchecked(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_vec_unchecked");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || generate_nibbles_random(size),
                |data| Nibbles::from_vec_unchecked(black_box(data)),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_first(c: &mut Criterion) {
    let mut group = c.benchmark_group("first");

    for size in SIZE_NIBBLES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| black_box(&data).first(),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_last(c: &mut Criterion) {
    let mut group = c.benchmark_group("last");

    for size in SIZE_NIBBLES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |data| black_box(&data).last(),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_starts_with(c: &mut Criterion) {
    let mut group = c.benchmark_group("starts_with");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || {
                    let nibbles = Nibbles::from_nibbles(generate_nibbles_random(size));
                    let prefix_len = size / 4;
                    let prefix = nibbles.slice(..prefix_len);
                    (nibbles, prefix)
                },
                |(data, prefix)| black_box(&data).starts_with(black_box(&prefix)),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_ends_with(c: &mut Criterion) {
    let mut group = c.benchmark_group("ends_with");

    for size in SIZE_NIBBLES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || {
                    let nibbles = Nibbles::from_nibbles(generate_nibbles_random(size));
                    let suffix_len = size / 4;
                    let suffix = nibbles.slice(size - suffix_len..);
                    (nibbles, suffix)
                },
                |(data, suffix)| black_box(&data).ends_with(black_box(&suffix)),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_truncate(c: &mut Criterion) {
    let mut group = c.benchmark_group("truncate");

    for size in SIZE_NIBBLES {
        let new_len = size / 2;

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |mut nib| {
                    nib.truncate(black_box(new_len));
                    nib
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

pub fn bench_clear(c: &mut Criterion) {
    let mut group = c.benchmark_group("clear");

    for size in SIZE_NIBBLES {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || Nibbles::from_nibbles(generate_nibbles_random(size)),
                |mut nib| {
                    nib.clear();
                    nib
                },
                criterion::BatchSize::SmallInput,
            )
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
                b.iter_batched(
                    || generate_bytes_random(size),
                    |bytes| unpack_naive(black_box(&bytes)),
                    criterion::BatchSize::SmallInput,
                )
            });

            let id = criterion::BenchmarkId::new("nybbles", size);
            g.bench_function(id, |b| {
                b.iter_batched(
                    || generate_bytes_random(size),
                    |bytes| Nibbles::unpack(black_box(&bytes)),
                    criterion::BatchSize::SmallInput,
                )
            });
        }
    }

    {
        let mut g = c.benchmark_group("pack");
        for size in SIZE_NIBBLES {
            g.throughput(criterion::Throughput::Elements(size as u64));

            let id = criterion::BenchmarkId::new("naive", size);
            g.bench_function(id, |b| {
                b.iter_batched(
                    || generate_nibbles_random(size),
                    |bytes| pack_naive(black_box(&bytes)),
                    criterion::BatchSize::SmallInput,
                )
            });

            let id = criterion::BenchmarkId::new("nybbles", size);
            g.bench_function(id, |b| {
                b.iter_batched(
                    || Nibbles::from_nibbles(generate_nibbles_random(size)),
                    |bytes| black_box(&bytes).pack(),
                    criterion::BatchSize::SmallInput,
                )
            });
        }
    }
}

fn generate_bytes_random(len: usize) -> Vec<u8> {
    let mut rng = thread_rng();
    (0..len).map(|_| rng.gen()).collect()
}

fn generate_nibbles_random(len: usize) -> Vec<u8> {
    let mut rng = thread_rng();
    (0..len).map(|_| rng.gen_range(0..16)).collect()
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
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .noise_threshold(0.20);
    targets = bench_from_nibbles, bench_pack, bench_unpack, bench_push, bench_slice,
              bench_join, bench_extend, bench_set_at, bench_get_byte, bench_common_prefix_length,
              bench_cmp, bench_clone, bench_increment, bench_pop, bench_from_vec_unchecked, bench_first,
              bench_last, bench_starts_with, bench_ends_with, bench_truncate, bench_clear,
              nibbles_benchmark
);
criterion_main!(benches);

#[test]
fn naive_equivalency() {
    for len in [0, 1, 2, 3, 4, 15, 16, 17, 31, 32, 33] {
        let bytes = generate_bytes_random(len);
        let nibbles = Nibbles::unpack(&bytes);
        assert_eq!(unpack_naive(&bytes)[..], nibbles[..]);
        assert_eq!(pack_naive(&nibbles[..])[..], nibbles.pack()[..]);
    }
}
