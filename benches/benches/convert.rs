use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    for &size in &SIZE_NIBBLES {
        bench_arbitrary_with(
            criterion,
            &format!("from_nibbles/{size}"),
            arb_vec(0u8..16, size).prop_map(|v| {
                let mut a = [0u8; 64];
                a[..v.len()].copy_from_slice(&v);
                a
            }),
            |a| Nibbles::from_nibbles(black_box(&a[..size])),
        );
    }

    for &size in &SIZE_BYTES {
        bench_arbitrary_with(
            criterion,
            &format!("unpack/{size}"),
            proptest::arbitrary::any::<[u8; 32]>(),
            |bytes| Nibbles::unpack(black_box(&bytes[..size])),
        );
    }

    for &size in &SIZE_BYTES {
        bench_arbitrary_with(
            criterion,
            &format!("pack/{size}"),
            nibbles_strategy(size * 2),
            |nibbles| black_box(&nibbles).pack(),
        );
    }

    for &size in &SIZE_BYTES {
        bench_unop(criterion, "pack_to", size * 2, |nibbles| {
            let mut buf = SmallVec::<[u8; 32]>::new();
            buf.resize(nibbles.len().div_ceil(2), 0);
            nibbles.pack_to(black_box(&mut buf));
            buf
        });
    }
}
