//! Verification script for the new property tests

use nybbles::Nibbles;

fn main() {
    println!("Testing Nibbles conversion methods manually...");

    // Test pack() functionality
    println!("\n1. Testing pack() method:");
    let nibbles = Nibbles::from_nibbles([0x0A, 0x0B, 0x0C, 0x0D]);
    let packed = nibbles.pack();
    println!("   Nibbles: {:?}", nibbles.as_slice());
    println!("   Packed: {:?}", &packed[..]);
    assert_eq!(&packed[..], &[0xAB, 0xCD]);
    println!("   ✓ Pack works correctly");

    // Test to_vec() functionality
    println!("\n2. Testing to_vec() method:");
    let vec_result = nibbles.to_vec();
    println!("   to_vec(): {:?}", vec_result);
    println!("   as_slice(): {:?}", nibbles.as_slice());
    assert_eq!(vec_result, nibbles.as_slice());
    println!("   ✓ to_vec() works correctly");

    // Test pack_to() functionality
    println!("\n3. Testing pack_to() method:");
    let mut buffer = vec![0u8; 2];
    nibbles.pack_to(&mut buffer);
    println!("   pack_to() result: {:?}", buffer);
    assert_eq!(buffer, vec![0xAB, 0xCD]);
    println!("   ✓ pack_to() works correctly");

    // Test roundtrip functionality
    println!("\n4. Testing roundtrip properties:");

    // pack -> unpack roundtrip
    let unpacked = Nibbles::unpack(&packed);
    println!("   Original: {:?}", nibbles.as_slice());
    println!("   Packed: {:?}", &packed[..]);
    println!("   Unpacked: {:?}", unpacked.as_slice());
    assert_eq!(unpacked.as_slice(), nibbles.as_slice());
    println!("   ✓ pack -> unpack roundtrip works");

    // to_vec -> from_vec roundtrip
    let from_vec_result = Nibbles::from_vec(vec_result);
    println!("   Original: {:?}", nibbles.as_slice());
    println!("   to_vec -> from_vec: {:?}", from_vec_result.as_slice());
    assert_eq!(from_vec_result.as_slice(), nibbles.as_slice());
    println!("   ✓ to_vec -> from_vec roundtrip works");

    // Test packed length invariant
    println!("\n5. Testing packed length invariant:");
    let test_cases = vec![
        vec![],
        vec![0x0A],
        vec![0x0A, 0x0B],
        vec![0x0A, 0x0B, 0x0C],
        vec![0x0A, 0x0B, 0x0C, 0x0D],
        vec![0x0A, 0x0B, 0x0C, 0x0D, 0x0E],
    ];

    for test_case in test_cases {
        let nibbles = Nibbles::from_nibbles(&test_case);
        let packed = nibbles.pack();
        let expected_len = test_case.len().div_ceil(2);
        println!(
            "   Nibbles len: {}, Expected packed len: {}, Actual packed len: {}",
            test_case.len(),
            expected_len,
            packed.len()
        );
        assert_eq!(packed.len(), expected_len);
    }
    println!("   ✓ Packed length invariant holds");

    // Test edge cases
    println!("\n6. Testing edge cases:");

    // Empty nibbles
    let empty = Nibbles::new();
    let empty_packed = empty.pack();
    let empty_vec = empty.to_vec();
    assert_eq!(empty_packed.len(), 0);
    assert_eq!(empty_vec.len(), 0);
    println!("   ✓ Empty nibbles handled correctly");

    // Odd length nibbles (padding test)
    let odd_nibbles = Nibbles::from_nibbles([0x0A, 0x0B, 0x0C]);
    let odd_packed = odd_nibbles.pack();
    let odd_unpacked = Nibbles::unpack(&odd_packed);
    println!("   Odd nibbles: {:?}", odd_nibbles.as_slice());
    println!("   Packed: {:?}", &odd_packed[..]);
    println!("   Unpacked: {:?}", odd_unpacked.as_slice());
    assert_eq!(&odd_packed[..], &[0xAB, 0xC0]); // Last nibble padded with 0
    assert_eq!(odd_unpacked.as_slice(), &[0x0A, 0x0B, 0x0C, 0x00]); // Padding added
    println!("   ✓ Odd length nibbles handled correctly");

    println!("\n🎉 All manual tests passed! The property tests should work correctly.");
}
