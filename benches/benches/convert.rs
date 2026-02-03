use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    for &size in &SIZE_NIBBLES {
        bench_arbitrary_with(
            criterion,
            &format!("from_nibbles/{size}"),
            arb_vec(0u8..16, size),
            |v| Nibbles::from_nibbles(black_box(&v)),
        );
    }

    for &size in &SIZE_BYTES {
        bench_arbitrary_with(criterion, &format!("unpack/{size}"), bytes_strategy(size), |bytes| {
            Nibbles::unpack(black_box(&bytes))
        });
    }

    for &size in &SIZE_BYTES {
        bench_arbitrary_with(
            criterion,
            &format!("pack/{size}"),
            bytes_strategy(size).prop_map(|b| Nibbles::unpack(&b)),
            |nibbles| black_box(&nibbles).pack(),
        );
    }

    for &size in &SIZE_BYTES {
        bench_arbitrary_with(
            criterion,
            &format!("pack_to/{size}"),
            bytes_strategy(size).prop_map(|b| Nibbles::unpack(&b)),
            |nibbles| {
                let mut buf = vec![0u8; nibbles.len().div_ceil(2)];
                nibbles.pack_to(black_box(&mut buf));
                buf
            },
        );
    }
}
