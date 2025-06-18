mod prelude;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nybbles::Nibbles;
use prelude::*;
use proptest::{prelude::Just, strategy::Strategy};
use std::time::Duration;

const SIZE_NIBBLES: [usize; 4] = [8, 16, 32, 64];
const SIZE_BYTES: [usize; 4] = [4, 8, 16, 32];

pub fn bench_from_nibbles(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("from_nibbles/{size}"),
            proptest::collection::vec(0u8..16, size),
            |data| Nibbles::from_nibbles(black_box(data)),
        );
    }
}

pub fn bench_from_vec_unchecked(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("from_vec_unchecked/{size}"),
            arbitrary_raw_nibbles(size),
            |data| Nibbles::from_vec_unchecked(black_box(data.clone())),
        );
    }
}

pub fn bench_pack(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(c, format!("pack/{size}"), arbitrary_nibbles(size), |nibbles| {
            black_box(nibbles).pack()
        });
    }

    for size in SIZE_BYTES {
        bench_arbitrary_with(
            c,
            format!("pack_to/{size}"),
            arbitrary_bytes(size)
                .prop_map(|nibbles| (vec![0; nibbles.len().div_ceil(2)], Nibbles::unpack(nibbles))),
            |(buffer, nibbles)| {
                black_box(nibbles).pack_to(black_box(&mut buffer.clone()));
            },
        );
    }
}

pub fn bench_unpack(c: &mut Criterion) {
    for size in SIZE_BYTES {
        bench_arbitrary_with(c, format!("unpack/{size}"), arbitrary_bytes(size), |data| {
            Nibbles::unpack(black_box(&data))
        });
    }
}

pub fn bench_push(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("push/{size}"),
            arbitrary_raw_nibbles(size),
            |raw_nibbles| {
                let mut nibbles = Nibbles::new();
                for nibble in raw_nibbles {
                    nibbles.push(black_box(*nibble));
                }
                nibbles
            },
        );
    }
}

pub fn bench_slice(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("slice/{size}"),
            arbitrary_nibbles(size)
                .prop_flat_map(|nibbles| {
                    let start = 0..(nibbles.len() - 1);
                    (Just(nibbles), start)
                })
                .prop_flat_map(|(nibbles, start)| {
                    let end = start..nibbles.len();
                    (Just(nibbles), Just(start), end)
                }),
            |(nibbles, start, end)| nibbles.slice(black_box(*start..*end)),
        );
    }
}

pub fn bench_join(c: &mut Criterion) {
    for &size in &SIZE_NIBBLES[..SIZE_NIBBLES.len() - 1] {
        bench_arbitrary_with(
            c,
            format!("join/{size}"),
            (arbitrary_nibbles(size), arbitrary_nibbles(size)),
            |(a, b)| a.join(black_box(b)),
        );
    }
}

pub fn bench_extend(c: &mut Criterion) {
    for &size in &SIZE_NIBBLES[..SIZE_NIBBLES.len() - 1] {
        bench_arbitrary_with(
            c,
            format!("extend/{size}"),
            (arbitrary_nibbles(size), arbitrary_nibbles(size)),
            |(base, extension)| {
                base.clone().extend_from_slice(black_box(extension));
            },
        );
    }
}

pub fn bench_set_at(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("set_at/{size}"),
            arbitrary_non_empty_nibbles(size).prop_flat_map(|nibbles| {
                let i = 0..nibbles.len();
                (Just(nibbles), i)
            }),
            |(nibbles, i)| {
                nibbles.clone().set_at(black_box(*i), black_box((i % 16) as u8));
            },
        );
    }
}

pub fn bench_get_byte(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        let strategy = arbitrary_non_empty_nibbles(size).prop_flat_map(|nibbles| {
            let i = 0..nibbles.len();
            (Just(nibbles), i)
        });

        bench_arbitrary_with(c, format!("get_byte/{size}"), strategy.clone(), |(nibbles, i)| {
            nibbles.get_byte(black_box(*i));
        });

        bench_arbitrary_with(
            c,
            format!("get_byte_unchecked/{size}"),
            strategy,
            |(nibbles, i)| unsafe {
                nibbles.get_byte_unchecked(black_box(*i));
            },
        );
    }
}

pub fn bench_common_prefix_length(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("common_prefix_length/{size}"),
            arbitrary_nibbles(size)
                .prop_flat_map(|nibbles| {
                    let prefix_size = 0..nibbles.len();
                    (Just(nibbles), prefix_size)
                })
                .prop_map(|(nibbles, prefix_size)| {
                    let prefix = nibbles.slice(..prefix_size);
                    (nibbles, prefix)
                }),
            |(nibbles, prefix)| nibbles.common_prefix_length(black_box(prefix)),
        );
    }
}

pub fn bench_cmp(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("cmp/{size}"),
            (arbitrary_nibbles(size), arbitrary_nibbles(size)),
            |(a, b)| a.cmp(black_box(b)),
        );
    }
}

pub fn bench_clone(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(c, format!("clone/{size}"), arbitrary_nibbles(size), |nibbles| {
            black_box(nibbles).clone()
        });
    }
}

pub fn bench_increment(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(c, format!("increment/{size}"), arbitrary_nibbles(size), |nibbles| {
            black_box(nibbles).increment()
        });
    }
}

pub fn bench_pop(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(c, format!("pop/{size}"), arbitrary_nibbles(size), |nibbles| {
            let mut nibbles = nibbles.clone();
            while nibbles.pop().is_some() {}
        });
    }
}

pub fn bench_first(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(c, format!("first/{size}"), arbitrary_nibbles(size), |nibbles| {
            black_box(&nibbles).first()
        });
    }
}

pub fn bench_last(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(c, format!("last/{size}"), arbitrary_nibbles(size), |nibbles| {
            black_box(&nibbles).last()
        });
    }
}

pub fn bench_starts_with(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("starts_with/{size}"),
            arbitrary_nibbles(size)
                .prop_flat_map(|nibbles| {
                    let prefix_size = 0..nibbles.len();
                    (Just(nibbles), prefix_size)
                })
                .prop_map(|(nibbles, prefix_size)| {
                    let prefix = nibbles.slice(..prefix_size);
                    (nibbles, prefix)
                }),
            |(nibbles, prefix)| nibbles.starts_with(black_box(prefix)),
        );
    }
}

pub fn bench_ends_with(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("ends_with/{size}"),
            arbitrary_nibbles(size)
                .prop_flat_map(|nibbles| {
                    let suffix_size = 0..nibbles.len();
                    (Just(nibbles), suffix_size)
                })
                .prop_map(|(nibbles, suffix_size)| {
                    let suffix = nibbles.slice(nibbles.len() - suffix_size..);
                    (nibbles, suffix)
                }),
            |(nibbles, suffix)| nibbles.ends_with(black_box(suffix)),
        );
    }
}

pub fn bench_truncate(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(
            c,
            format!("truncate/{size}"),
            arbitrary_nibbles(size).prop_flat_map(|nibbles| {
                let new_len = 0..nibbles.len();
                (Just(nibbles), new_len)
            }),
            |(nibbles, new_len)| {
                nibbles.clone().truncate(black_box(*new_len));
            },
        );
    }
}

pub fn bench_clear(c: &mut Criterion) {
    for size in SIZE_NIBBLES {
        bench_arbitrary_with(c, format!("clear/{size}"), arbitrary_nibbles(size), |nibbles| {
            black_box(nibbles.clone()).clear()
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .noise_threshold(0.20);
    targets = bench_from_nibbles, bench_pack, bench_unpack, bench_push, bench_slice,
              bench_join, bench_extend, bench_set_at, bench_get_byte, bench_common_prefix_length,
              bench_cmp, bench_clone, bench_increment, bench_pop, bench_from_vec_unchecked,
              bench_first, bench_last, bench_starts_with, bench_ends_with, bench_truncate,
              bench_clear
);
criterion_main!(benches);
