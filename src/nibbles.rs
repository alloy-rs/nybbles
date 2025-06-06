use core::{
    cmp::Ordering,
    fmt,
    mem::MaybeUninit,
    ops::{Bound, Index, RangeBounds},
    slice,
};
use ruint::aliases::U256;
use smallvec::SmallVec;

#[cfg(not(feature = "nightly"))]
#[allow(unused_imports)]
use core::convert::{identity as likely, identity as unlikely};
#[cfg(feature = "nightly")]
#[allow(unused_imports)]
use core::intrinsics::{likely, unlikely};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

type Repr = U256;

/// This array contains 65 bitmasks used in [`Nibbles::slice`].
///
/// Each mask is a [`U256`] where:
/// - Index 0 is just 0 (no bits set)
/// - Index 1 has the lowest 4 bits set (one nibble)
/// - Index 2 has the lowest 8 bits set (two nibbles)
/// - ...and so on
/// - Index 64 has all bits set ([`U256::MAX`])
const SLICE_MASKS: [U256; 65] = {
    let mut masks = [U256::ZERO; 65];
    let mut i = 0;
    while i <= 64 {
        masks[i] = U256::MAX.wrapping_shr(256 - i * 4);
        i += 1;
    }
    masks
};

/// Structure representing a sequence of nibbles.
///
/// A nibble is a 4-bit value, and this structure is used to store the nibble sequence representing
/// the keys in a Merkle Patricia Trie (MPT).
/// Using nibbles simplifies trie operations and enables consistent key representation in the MPT.
///
/// # Internal representation
///
/// The internal representation is currently a [`SmallVec`] that stores one nibble per byte. Nibbles
/// are stored inline (on the stack) up to a length of 64 nibbles, or 32 unpacked bytes. This means
/// that each byte has its upper 4 bits set to zero and the lower 4 bits representing the nibble
/// value.
///
/// This is enforced in the public API, but it is possible to create invalid [`Nibbles`] instances
/// using methods suffixed with `_unchecked`. These methods are not marked as `unsafe` as they
/// are not memory-unsafe, but creating invalid values will cause unexpected behavior in other
/// methods, and users should exercise caution when using them.
///
/// # Examples
///
/// ```
/// use nybbles::Nibbles;
///
/// let bytes = [0xAB, 0xCD];
/// let nibbles = Nibbles::unpack(&bytes);
/// assert_eq!(nibbles, Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]));
/// assert_eq!(nibbles[..], [0x0A, 0x0B, 0x0C, 0x0D]);
///
/// let packed = nibbles.pack();
/// assert_eq!(&packed[..], &bytes[..]);
/// ```
#[repr(C)] // We want to preserve the order of fields in the memory layout.
#[derive(Default, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct Nibbles {
    /// Nibbles length.
    // This field goes first, because the derived implementation of `PartialEq` compares the fields
    // in order, so we can short-circuit the comparison if the `length` field differs.
    pub(crate) length: u8,
    /// The nibbles themselves, stored as a 256-bit unsigned integer.
    pub(crate) nibbles: U256,
}

impl fmt::Debug for Nibbles {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "Nibbles(0x)")
        } else {
            write!(f, "Nibbles(0x{:0width$x})", self.nibbles, width = self.length as usize)
        }
    }
}

// Deriving [`Ord`] for [`Nibbles`] is not correct, because they will be compared as unsigned
// integers without accounting for length. This is incorrect, because `0x1` should be considered
// greater than `0x02`.
impl Ord for Nibbles {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.length.cmp(&other.length) {
            Ordering::Equal => self.nibbles.cmp(&other.nibbles),
            Ordering::Greater => {
                let shift = ((self.length - other.length) as usize) * 4;
                let cmp = self.nibbles.wrapping_shr(shift).cmp(&other.nibbles);
                if cmp == Ordering::Equal {
                    Ordering::Greater
                } else {
                    cmp
                }
            }
            Ordering::Less => {
                let shift = ((other.length - self.length) as usize) * 4;
                let cmp = self.nibbles.cmp(&other.nibbles.wrapping_shr(shift));
                if cmp == Ordering::Equal {
                    Ordering::Less
                } else {
                    cmp
                }
            }
        }
    }
}

impl PartialOrd for Nibbles {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Index<usize> for Nibbles {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        /// List of possible nibbles to return static references.
        static NIBBLES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        &NIBBLES[self.get_unchecked(index) as usize]
    }
}

impl From<Nibbles> for Vec<u8> {
    #[inline]
    fn from(value: Nibbles) -> Self {
        let mut nibbles = Self::with_capacity(value.len());
        for i in 0..value.len() {
            nibbles.push(value.get_unchecked(i));
        }
        nibbles
    }
}

#[cfg(feature = "rlp")]
impl alloy_rlp::Encodable for Nibbles {
    #[inline]
    fn length(&self) -> usize {
        alloy_rlp::Encodable::length(self.as_slice())
    }

    #[inline]
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        alloy_rlp::Encodable::encode(self.as_slice(), out)
    }
}

#[cfg(feature = "arbitrary")]
impl proptest::arbitrary::Arbitrary for Nibbles {
    type Parameters = proptest::collection::SizeRange;
    type Strategy = proptest::strategy::Map<
        proptest::collection::VecStrategy<core::ops::RangeInclusive<u8>>,
        fn(Vec<u8>) -> Self,
    >;

    #[inline]
    fn arbitrary_with(size: proptest::collection::SizeRange) -> Self::Strategy {
        use proptest::prelude::*;
        proptest::collection::vec(0x0..=0xf, size).prop_map(Self::from_nibbles_unchecked)
    }
}

impl Nibbles {
    /// Creates a new empty [`Nibbles`] instance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::new();
    /// assert_eq!(nibbles.len(), 0);
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self { length: 0, nibbles: U256::ZERO }
    }

    /// Build from *any* iterator of nibble values – slices, `Vec`, ranges…
    #[inline]
    #[track_caller]
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        let mut packed = Self::default();
        for n in iter {
            assert!(n <= 0x0f, "invalid nibble {n:#x}");
            packed.nibbles = (packed.nibbles << 4) | U256::from(n);
            packed.length += 1;
        }
        packed
    }

    /// Same as `from_iter`, but skips the validity check.
    #[inline]
    pub fn from_iter_unchecked<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        let mut packed = Self::default();
        for n in iter {
            packed.nibbles = (packed.nibbles << 4) | U256::from(n & 0x0f);
            packed.length += 1;
        }
        packed
    }

    /// Creates a new [`Nibbles`] instance by copying the given bytes.
    ///
    /// # Panics
    ///
    /// Panics if the any of the bytes is not a valid nibble (`0..=0x0f`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]);
    /// assert_eq!(nibbles[..], [0x0A, 0x0B, 0x0C, 0x0D]);
    /// ```
    ///
    /// Invalid values will cause panics:
    ///
    /// ```should_panic
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_nibbles(&[0xFF]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn from_nibbles<T: AsRef<[u8]>>(nibbles: T) -> Self {
        let bytes = nibbles.as_ref();
        check_nibbles(bytes);
        Self::from_iter_unchecked(bytes.iter().copied())
    }

    /// Creates a new [`Nibbles`] instance by copying the given bytes, without checking their
    /// validity.
    ///
    /// This will not unpack the bytes into nibbles, and will instead store the bytes as-is.
    ///
    /// Note that it is possible to create a [`Nibbles`] instance with invalid nibble values (i.e.
    /// values greater than 0xf) using this method. See [the type docs](Self) for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_nibbles_unchecked(&[0x0A, 0x0B, 0x0C, 0x0D]);
    /// assert_eq!(nibbles[..], [0x0A, 0x0B, 0x0C, 0x0D]);
    ///
    /// // Invalid value!
    /// let nibbles = Nibbles::from_nibbles_unchecked(&[0xFF]);
    /// assert_eq!(nibbles[..], [0xFF]);
    /// ```
    #[inline]
    pub fn from_nibbles_unchecked<T: AsRef<[u8]>>(nibbles: T) -> Self {
        Self::from_iter_unchecked(nibbles.as_ref().iter().copied())
    }

    /// Creates a new [`Nibbles`] instance from a byte vector, without checking its validity.
    ///
    /// This will not unpack the bytes into nibbles, and will instead store the bytes as-is.
    ///
    /// Note that it is possible to create a [`Nibbles`] instance with invalid nibble values (i.e.
    /// values greater than 0xf) using this method. See [the type docs](Self) for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_vec_unchecked(vec![0x0A, 0x0B, 0x0C, 0x0D]);
    /// assert_eq!(nibbles[..], [0x0A, 0x0B, 0x0C, 0x0D]);
    /// ```
    ///
    /// Invalid values will cause panics:
    ///
    /// ```should_panic
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_vec(vec![0xFF]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn from_vec(vec: Vec<u8>) -> Self {
        check_nibbles(&vec);
        Self::from_vec_unchecked(vec)
    }

    /// Creates a new [`Nibbles`] instance from a byte vector, without checking its validity.
    ///
    /// Note that it is possible to create a [`Nibbles`] instance with invalid nibble values (i.e.
    /// values greater than 0xf) using this method. See [the type docs](Self) for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_vec_unchecked(vec![0x0A, 0x0B, 0x0C, 0x0D]);
    /// assert_eq!(nibbles[..], [0x0A, 0x0B, 0x0C, 0x0D]);
    ///
    /// // Invalid value!
    /// let nibbles = Nibbles::from_vec_unchecked(vec![0xFF]);
    /// assert_eq!(nibbles[..], [0xFF]);
    /// ```
    #[inline]
    pub fn from_vec_unchecked(vec: Vec<u8>) -> Self {
        Self::from_nibbles_unchecked(vec)
    }

    /// Converts a byte slice into a [`Nibbles`] instance containing the nibbles (half-bytes or 4
    /// bits) that make up the input byte data.
    ///
    /// # Panics
    ///
    /// Panics if the length of the input is greater than `usize::MAX / 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::unpack(&[0xAB, 0xCD]);
    /// assert_eq!(nibbles[..], [0x0A, 0x0B, 0x0C, 0x0D]);
    /// ```
    #[inline]
    pub fn unpack<T: AsRef<[u8]>>(data: T) -> Self {
        let bytes = data.as_ref();
        let mut packed = Self::default();

        // Convert each byte to two nibbles
        for &byte in bytes {
            let high_nibble = byte >> 4;
            let low_nibble = byte & 0x0F;

            // TODO: optimize
            // Pack high nibble
            packed.nibbles = (packed.nibbles << 4) | U256::from(high_nibble);
            packed.length += 1;

            // Pack low nibble
            packed.nibbles = (packed.nibbles << 4) | U256::from(low_nibble);
            packed.length += 1;
        }

        packed
    }

    /// Unpacks into the given slice rather than allocating a new [`Nibbles`] instance.
    #[inline]
    pub fn unpack_to(data: &[u8], out: &mut [u8]) {
        assert!(out.len() >= data.len() * 2);
        // SAFETY: asserted length.
        unsafe {
            let out = slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len());
            Self::unpack_to_unchecked(data, out)
        }
    }

    /// Unpacks into the given slice rather than allocating a new [`Nibbles`] instance.
    ///
    /// # Safety
    ///
    /// `out` must be valid for at least `data.len() * 2` bytes.
    #[inline]
    pub unsafe fn unpack_to_unchecked(data: &[u8], out: &mut [MaybeUninit<u8>]) {
        debug_assert!(out.len() >= data.len() * 2);
        let ptr = out.as_mut_ptr().cast::<u8>();
        for (i, &byte) in data.iter().enumerate() {
            ptr.add(i * 2).write(byte >> 4);
            ptr.add(i * 2 + 1).write(byte & 0x0f);
        }
    }

    /// Packs the nibbles into the given slice.
    ///
    /// This method combines each pair of consecutive nibbles into a single byte,
    /// effectively reducing the size of the data by a factor of two.
    /// If the number of nibbles is odd, the last nibble is shifted left by 4 bits and
    /// added to the packed byte vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]);
    /// assert_eq!(nibbles.pack()[..], [0xAB, 0xCD]);
    /// ```
    #[inline]
    pub fn pack(&self) -> SmallVec<[u8; 32]> {
        let packed_len = (self.len() + 1) / 2;
        unsafe { smallvec_with(packed_len, |out| self.pack_to_unchecked2(out)) }
    }

    /// Packs the nibbles into the given slice.
    ///
    /// See [`pack`](Self::pack) for more information.
    ///
    /// # Panics
    ///
    /// Panics if the slice is not at least `(self.len() + 1) / 2` bytes long.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]);
    /// let mut packed = [0; 2];
    /// nibbles.pack_to(&mut packed);
    /// assert_eq!(packed[..], [0xAB, 0xCD]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn pack_to(&self, out: &mut [u8]) {
        pack_to(self, out);
    }

    /// Packs the nibbles into the given pointer.
    ///
    /// See [`pack`](Self::pack) for more information.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for at least `(self.len() + 1) / 2` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]);
    /// let mut packed = [0; 2];
    /// // SAFETY: enough capacity.
    /// unsafe { nibbles.pack_to_unchecked(packed.as_mut_ptr()) };
    /// assert_eq!(packed[..], [0xAB, 0xCD]);
    /// ```
    #[inline]
    #[deprecated = "prefer using `pack_to` or `pack_to_unchecked2` instead"]
    pub unsafe fn pack_to_unchecked(&self, ptr: *mut u8) {
        self.pack_to_unchecked2(slice::from_raw_parts_mut(ptr.cast(), (self.len() + 1) / 2));
    }

    /// Packs the nibbles into the given slice without checking its length.
    ///
    /// # Safety
    ///
    /// `out` must be valid for at least `(self.len() + 1) / 2` bytes.
    #[inline]
    pub unsafe fn pack_to_unchecked2(&self, out: &mut [MaybeUninit<u8>]) {
        pack_to_unchecked(self, out);
    }

    /// Gets the byte at the given index by combining two consecutive nibbles.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]);
    /// assert_eq!(nibbles.get_byte(0), Some(0xAB));
    /// assert_eq!(nibbles.get_byte(1), Some(0xBC));
    /// assert_eq!(nibbles.get_byte(2), Some(0xCD));
    /// assert_eq!(nibbles.get_byte(3), None);
    /// ```
    pub fn get_byte(&self, i: usize) -> Option<u8> {
        // Abort early if out of range
        if i >= self.byte_len() {
            return None;
        }

        // Distance (in windows) from the most-significant byte
        let pos_from_back = self.byte_len() - 1 - i;

        // 8 bytes per 64-bit limb
        let limb = pos_from_back / 8;
        let offset = (pos_from_back % 8) * 8;

        let word = *self.nibbles.as_limbs().get(limb)?;
        Some(((word >> offset) & 0xFF) as u8)
    }

    /// Gets the byte at the given index by combining two consecutive nibbles.
    ///
    /// # Safety
    ///
    /// `i..i + 1` must be in range.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]);
    /// // SAFETY: in range.
    /// unsafe {
    ///     assert_eq!(nibbles.get_byte_unchecked(0), 0xAB);
    ///     assert_eq!(nibbles.get_byte_unchecked(1), 0xBC);
    ///     assert_eq!(nibbles.get_byte_unchecked(2), 0xCD);
    /// }
    /// ```
    pub unsafe fn get_byte_unchecked(&self, i: usize) -> u8 {
        debug_assert!(i < self.byte_len());
        let pos_from_back = self.byte_len() - 1 - i;
        let limb = pos_from_back / 8;
        let offset = (pos_from_back % 8) * 8;
        let word = self.nibbles.as_limbs()[limb];
        ((word >> offset) & 0xFF) as u8
    }

    /// Increments the nibble sequence by one.
    #[inline]
    pub fn increment(&self) -> Option<Self> {
        let mut incremented = self.clone();
        incremented.nibbles.checked_add(U256::ONE)?;
        if self.nibbles.leading_zeros() / 4 != incremented.nibbles.leading_zeros() / 4 {
            incremented.length += 1;
        }
        Some(incremented)
    }

    /// The last element of the hex vector is used to determine whether the nibble sequence
    /// represents a leaf or an extension node. If the last element is 0x10 (16), then it's a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.last() == Some(16)
    }

    /// Returns `true` if this nibble sequence starts with the given prefix.
    pub const fn starts_with(&self, other: &Self) -> bool {
        // If other is empty, it's a prefix of any sequence
        if other.is_empty() {
            return true;
        }

        // If other is longer than self, it can't be a prefix
        if other.len() > self.len() {
            return false;
        }

        let mut i = 0;
        while i < other.len() {
            if self.get_unchecked(i) != other.get_unchecked(i) {
                return false;
            }
            i += 1;
        }

        true
    }

    /// Returns `true` if this nibble sequence starts with the given prefix.
    pub const fn has_prefix(&self, other: &[u8]) -> bool {
        // If other is empty, it's a prefix of any sequence
        if other.is_empty() {
            return true;
        }

        // If other is longer than self, it can't be a prefix
        if other.len() > self.len() {
            return false;
        }

        let mut i = 0;
        while i < other.len() {
            if self.get_unchecked(i) != other[i] {
                return false;
            }
            i += 1;
        }

        true
    }

    /// Returns the nibble at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub const fn get_unchecked(&self, i: usize) -> u8 {
        // How far from the most-significant nibble?
        let pos_from_back = self.len() - 1 - i; // 0-based from MSB
        let limb = pos_from_back / 16; // 16 nibbles per u64 limb
        let offset = (pos_from_back % 16) * 4; // Offset bits within that limb, so we get the one we're interested in

        let word = self.nibbles.as_limbs()[limb];
        ((word >> offset) & 0xF) as u8
    }

    /// Returns the nibble at the given index.
    pub fn get(&self, i: usize) -> Option<u8> {
        // How far from the most-significant nibble?
        let pos_from_back = self.len().checked_sub(1 + i)?; // 0-based from MSB
        let limb = pos_from_back / 16; // 16 nibbles per u64 limb
        let offset = (pos_from_back % 16) * 4; // Offset bits within that limb, so we get the one we're interested in

        let word = self.nibbles.as_limbs()[limb];
        Some(((word >> offset) & 0xF) as u8)
    }

    /// Sets the nibble at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds, or if `value` is not a valid nibble (`0..=0x0f`).
    #[inline]
    #[track_caller]
    pub fn set_at(&mut self, i: usize, value: u8) {
        assert!(value <= 0xf);
        self.set_at_unchecked(i, value);
    }

    /// Sets the nibble at the given index, without checking its validity.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    #[track_caller]
    pub fn set_at_unchecked(&mut self, i: usize, value: u8) {
        assert!(i < self.length as usize, "index out of bounds");
        let pos_from_back = self.len() - 1 - i; // 0-based from MSB
        let limb = pos_from_back / 16; // 16 nibbles per u64 limb
        let offset = (pos_from_back % 16) * 4; // Offset bits within that limb, so we get the one we're interested in

        // SAFETY: We checked that index is within bounds, so the limb is also within bounds
        let word = unsafe { self.nibbles.as_limbs_mut().get_unchecked_mut(limb) };
        *word &= !(0xF << offset); // Clear old nibble
        *word |= (value as u64) << offset; // Set new nibble
    }

    /// Returns the first nibble of this nibble sequence.
    pub fn first(&self) -> Option<u8> {
        if self.length == 0 {
            None
        } else {
            Some(self.get_unchecked(0))
        }
    }

    /// Returns the last nibble of this nibble sequence.
    pub fn last(&self) -> Option<u8> {
        if self.length == 0 {
            None
        } else {
            Some(self.get_unchecked(self.length as usize - 1))
        }
    }

    /// Returns the length of the common prefix between this nibble sequence and the given.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let a = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
    /// let b = &[0x0A, 0x0B, 0x0C, 0x0E];
    /// assert_eq!(a.common_prefix_length(b), 3);
    /// ```
    pub fn common_prefix_length(&self, other: &Self) -> usize {
        let min_len = if self.len() < other.len() { self.len() } else { other.len() };
        let mut i = 0;
        while i < min_len {
            if self.get_unchecked(i) != other.get_unchecked(i) {
                return i;
            }
            i += 1;
        }
        i

        // TODO: the optimized implementation below fails the last test case

        // const fn count_equal_nibbles(self_limb: u64, other_limb: u64) -> usize {
        //     // Pad both limbs with trailing zeros to the same effective length
        //     let lhs_bit_len = u64::BITS - self_limb.leading_zeros(); // Effective bit length of
        // the left limb     let rhs_bit_len = u64::BITS - other_limb.leading_zeros(); //
        // Effective bit length of the right limb     let diff = lhs_bit_len as isize -
        // rhs_bit_len as isize; // Difference in bit lengths     let (lhs, rhs) = if diff <
        // 0 {         (self_limb << -diff, other_limb)
        //     } else {
        //         (self_limb, other_limb << diff)
        //     }; // Pad one of the limbs

        //     // Count equal leading bits
        //     let lz_or = (lhs | rhs).leading_zeros(); // Leading zeros common to both limbs
        //     let skip = lz_or & !0b11u32; // Leading zeros common to both limbs, rounded down to
        // the nearest nibble     let lz_xor = (lhs ^ rhs).leading_zeros(); // Leading bits
        // common to both limbs     (lz_xor - skip) as usize / 4
        // }

        // let self_bit_len = self.bit_len();
        // let other_bit_len = other.bit_len();

        // if self_bit_len == 0 || other_bit_len == 0 {
        //     return 0;
        // }

        // let min_bit_len = if self_bit_len < other_bit_len { self_bit_len } else { other_bit_len
        // };

        // // Number of whole limbs
        // let full_limbs = min_bit_len / 64;

        // let self_limbs = self.nibbles.as_limbs();
        // let other_limbs = other.nibbles.as_limbs();
        // let mut common_nibbles = 0;

        // // Walk from MS-limb to LS-limb
        // let mut i = full_limbs;
        // while i > 0 {
        //     i -= 1;
        //     if self_limbs[i] == other_limbs[i] {
        //         common_nibbles += 16;
        //     } else {
        //         // First differing limb – count equal nibbles inside it
        //         common_nibbles += count_equal_nibbles(self_limbs[i], other_limbs[i]);
        //         return common_nibbles;
        //     }
        // }

        // if min_bit_len % 64 == 0 {
        //     return common_nibbles;
        // }

        // common_nibbles + count_equal_nibbles(self_limbs[0], other_limbs[0])
    }

    /// Returns the total number of bytes in this [`Nibbles`].
    #[inline(always)]
    const fn byte_len(&self) -> usize {
        self.length as usize / 2
    }

    /// Returns the total number of bits in this [`Nibbles`].
    #[inline(always)]
    const fn bit_len(&self) -> usize {
        self.length as usize * 4
    }

    /// Returns `true` if this [`Nibbles`] is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the total number of nibbles in this [`Nibbles`].
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.length as usize
    }

    /// Returns a reference to the underlying byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.nibbles.as_le_slice()
    }

    /// Returns a mutable reference to the underlying byte slice.
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    #[inline]
    pub unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [u8] {
        self.nibbles.as_le_slice_mut()
    }

    /// Returns a mutable reference to the underlying [`Repr`].
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    #[inline]
    pub fn as_mut_uint_unchecked(&mut self) -> &mut Repr {
        &mut self.nibbles
    }

    /// Creates new nibbles containing the nibbles in the specified range `[start, end)`
    /// without checking bounds.
    ///
    /// # Safety
    ///
    /// This method does not verify that the provided range is valid for this nibble sequence.
    /// The caller must ensure that `start <= end` and `end <= self.len()`.
    #[inline]
    pub const fn slice_unchecked(&self, start: usize, end: usize) -> Self {
        // Fast path for empty slice
        if start == end {
            return Self::new();
        }

        // Fast path for full slice
        let slice_to_end = end == self.len();
        if start == 0 && slice_to_end {
            return *self;
        }

        let nibble_len = end - start;

        // Shift so that the first requested nibble becomes the least significant one
        let shifted = if slice_to_end {
            self.nibbles
        } else {
            let shift = (self.len() - end) * 4;
            self.nibbles.wrapping_shr(shift)
        };
        // Mask out everything to the left of the slice
        let nibbles = shifted.bitand(SLICE_MASKS[nibble_len]);

        Self { length: nibble_len as u8, nibbles }
    }

    /// Creates new nibbles containing the nibbles in the specified range.
    ///
    /// # Panics
    ///
    /// This method will panic if the range is out of bounds for this nibble sequence.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        // Determine the start and end nibble indices from the range bounds
        let start = match range.start_bound() {
            Bound::Included(&idx) => idx,
            Bound::Excluded(&idx) => idx + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&idx) => idx + 1,
            Bound::Excluded(&idx) => idx,
            Bound::Unbounded => self.len(),
        };
        assert!(start <= end, "Cannot slice with a start index greater than the end index");
        assert!(
            end <= self.len(),
            "Cannot slice with an end index greater than the length of the nibbles"
        );

        self.slice_unchecked(start, end)
    }

    /// Join two nibbles together.
    #[inline]
    pub fn join(&self, other: &Self) -> Self {
        let mut new = self.clone();
        if other.is_empty() {
            return new;
        }

        new.nibbles = new.nibbles.wrapping_shl(other.bit_len()).bitor(other.nibbles);
        new.length += other.length;
        new
    }

    /// Pushes a nibble to the end of the current nibbles.
    ///
    /// # Panics
    ///
    /// Panics if the nibble is not a valid nibble (`0..=0x0f`).
    #[inline]
    #[track_caller]
    pub fn push(&mut self, nibble: u8) {
        assert!(nibble <= 0xf);
        self.push_unchecked(nibble);
    }

    /// Pushes a nibble to the end of the current nibbles without checking its validity.
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    pub fn push_unchecked(&mut self, nibble: u8) {
        if self.length > 0 {
            self.nibbles = self.nibbles.wrapping_shl(4);
        }
        if nibble > 0 {
            self.nibbles = self.nibbles.bitor(U256::from_limbs([(nibble & 0x0F) as u64, 0, 0, 0]));
        }
        self.length += 1;
    }

    /// Pops a nibble from the end of the current nibbles.
    pub fn pop(&mut self) -> Option<u8> {
        if self.length == 0 {
            return None;
        }
        let nibble = self.get_unchecked(self.length as usize - 1);
        self.nibbles = self.nibbles.wrapping_shr(4);
        self.length -= 1;
        Some(nibble)
    }

    /// Extend the current nibbles with another nibbles.
    #[inline]
    pub fn extend_from_slice(&mut self, other: &Nibbles) {
        if other.is_empty() {
            return;
        }

        self.nibbles = self.nibbles.wrapping_shl(other.bit_len()).bitor(other.nibbles);
        self.length += other.length;
    }

    /// Extend the current nibbles with another byte slice.
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    #[inline]
    pub fn extend_from_slice_unchecked(&mut self, other: &[u8]) {
        if other.is_empty() {
            return;
        }

        self.nibbles = self.nibbles.wrapping_shl(other.len() * 8).bitor(U256::from_le_slice(other));
        self.length += other.len() as u8;
    }

    /// Truncates the current nibbles to the given length.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        assert!(
            new_len <= self.len(),
            "Cannot truncate to a length greater than the current length"
        );
        *self = self.slice_unchecked(0, new_len);
    }

    /// Clears the current nibbles.
    #[inline]
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

/// Packs the nibbles into the given slice.
///
/// See [`Nibbles::pack`] for more information.
///
/// # Panics
///
/// Panics if the slice is not at least `(self.len() + 1) / 2` bytes long.
///
/// # Examples
///
/// ```
/// # use nybbles::Nibbles;
/// let nibbles = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]);
/// let mut packed = [0; 2];
/// nibbles.pack_to(&mut packed);
/// assert_eq!(packed[..], [0xAB, 0xCD]);
/// ```
#[inline]
pub fn pack_to(nibbles: &Nibbles, out: &mut [u8]) {
    assert!(out.len() >= (nibbles.len() + 1) / 2);
    // SAFETY: asserted length.
    unsafe {
        let out = slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len());
        pack_to_unchecked(nibbles, out)
    }
}

/// Packs the nibbles into the given slice without checking its length.
///
/// # Safety
///
/// `out` must be valid for at least `(self.len() + 1) / 2` bytes.
#[inline]
pub unsafe fn pack_to_unchecked(nibbles: &Nibbles, out: &mut [MaybeUninit<u8>]) {
    let len = nibbles.len();
    debug_assert!(out.len() >= (len + 1) / 2);
    let ptr = out.as_mut_ptr().cast::<u8>();
    let mut i = 0;
    while i < len {
        let hi = nibbles.get_unchecked(i) << 4;
        let lo = if i + 1 < len { nibbles.get_unchecked(i + 1) } else { 0 };
        ptr.add(i / 2).write(hi | lo);
        i += 2;
    }
}

/// Initializes a smallvec with the given length and a closure that initializes the buffer.
///
/// Optimized version of `SmallVec::with_capacity` + `f()` + `.set_len`.
///
/// # Safety
///
/// The closure must fully initialize the buffer with the given length.
#[inline]
pub unsafe fn smallvec_with<const N: usize>(
    len: usize,
    f: impl FnOnce(&mut [MaybeUninit<u8>]),
) -> SmallVec<[u8; N]> {
    let mut buf = smallvec_with_len::<N>(len);
    f(unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), len) });
    buf
}

#[inline]
#[allow(clippy::uninit_vec)]
unsafe fn smallvec_with_len<const N: usize>(len: usize) -> SmallVec<[u8; N]> {
    if likely(len <= N) {
        SmallVec::from_buf_and_len_unchecked(MaybeUninit::<[u8; N]>::uninit(), len)
    } else {
        let mut vec = Vec::with_capacity(len);
        unsafe { vec.set_len(len) };
        SmallVec::from_vec(vec)
    }
}

#[inline]
#[track_caller]
fn check_nibbles(nibbles: &[u8]) {
    if !valid_nibbles(nibbles) {
        panic_invalid_nibbles();
    }
}

fn valid_nibbles(nibbles: &[u8]) -> bool {
    nibbles.iter().all(|&nibble| nibble <= 0xf)
}

#[cold]
#[track_caller]
const fn panic_invalid_nibbles() -> ! {
    panic!("attempted to create invalid nibbles");
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn pack_nibbles() {
        let tests = [
            (&[][..], &[][..]),
            (&[0xa], &[0xa0]),
            (&[0xa, 0x0], &[0xa0]),
            (&[0xa, 0xb], &[0xab]),
            (&[0xa, 0xb, 0x2], &[0xab, 0x20]),
            (&[0xa, 0xb, 0x2, 0x0], &[0xab, 0x20]),
            (&[0xa, 0xb, 0x2, 0x7], &[0xab, 0x27]),
        ];
        for (input, expected) in tests {
            assert!(valid_nibbles(input));
            let nibbles = Nibbles::from_nibbles(input);
            let encoded = nibbles.pack();
            assert_eq!(
                &encoded[..],
                expected,
                "input: {:x?}, expected: {:x?}, got: {:x?}",
                input,
                expected,
                encoded
            );
        }
    }

    #[test]
    fn at() {
        for len in 0..64 {
            let raw = (0..16).cycle().take(len).collect::<Vec<u8>>();
            let nibbles = Nibbles::from_nibbles(&raw);
            for i in 0..len {
                assert_eq!(nibbles.get_unchecked(i), raw[i]);
            }
        }
    }

    #[test]
    fn push_pop() {
        let mut nibbles = Nibbles::new();
        nibbles.push(0x0A);
        assert_eq!(nibbles.get_unchecked(0), 0x0A);
        assert_eq!(nibbles.len(), 1);

        assert_eq!(nibbles.pop(), Some(0x0A));
        assert_eq!(nibbles.len(), 0);
    }

    #[test]
    fn get_byte_max() {
        let nibbles = Nibbles::from_nibbles([0x0A, 0x0B, 0x0C, 0x0D]);
        assert_eq!(nibbles.get_byte(usize::MAX), None);
    }

    #[test]
    fn clone() {
        let a = Nibbles::from_nibbles([1, 2, 3]);
        #[allow(clippy::redundant_clone)]
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn ord() {
        // Test empty nibbles
        let nibbles1 = Nibbles::default();
        let nibbles2 = Nibbles::default();
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Equal);

        // Test with same nibbles
        let nibbles1 = Nibbles::unpack([0x12, 0x34]);
        let nibbles2 = Nibbles::unpack([0x12, 0x34]);
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Equal);

        // Test with different lengths
        let short = Nibbles::unpack([0x12]);
        let long = Nibbles::unpack([0x12, 0x34]);
        assert_eq!(short.cmp(&long), Ordering::Less);

        // Test with common prefix but different values
        let nibbles1 = Nibbles::unpack([0x12, 0x34]);
        let nibbles2 = Nibbles::unpack([0x12, 0x35]);
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Less);

        // Test with differing first byte
        let nibbles1 = Nibbles::unpack([0x12, 0x34]);
        let nibbles2 = Nibbles::unpack([0x13, 0x34]);
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Less);

        // Test with odd length nibbles
        let nibbles1 = Nibbles::unpack([0x1]);
        let nibbles2 = Nibbles::unpack([0x2]);
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Less);

        // Test with odd and even length nibbles
        let odd = Nibbles::unpack([0x1]);
        let even = Nibbles::unpack([0x12]);
        assert_eq!(odd.cmp(&even), Ordering::Less);

        // Test with longer sequences
        let nibbles1 = Nibbles::unpack([0x12, 0x34, 0x56, 0x78]);
        let nibbles2 = Nibbles::unpack([0x12, 0x34, 0x56, 0x79]);
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Less);

        let nibbles1 = Nibbles::from_nibbles([0x0, 0x0]);
        let nibbles2 = Nibbles::from_nibbles([0x1]);
        // Sanity check for nybbles
        assert_eq!(Nibbles::from(nibbles1).cmp(&Nibbles::from(nibbles2)), Ordering::Less);
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Less);

        let nibbles1 = Nibbles::from_nibbles([0x1]);
        let nibbles2 = Nibbles::from_nibbles([0x0, 0x2]);
        println!("nibbles1: {:x}, nibbles2: {:x}", nibbles1.nibbles, nibbles2.nibbles);
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Greater);
    }

    #[test]
    fn starts_with() {
        let nibbles = Nibbles::from_nibbles([1, 2, 3, 4]);

        // Test empty nibbles
        let empty = Nibbles::default();
        assert!(nibbles.starts_with(&empty));
        assert!(empty.starts_with(&empty));
        assert!(!empty.starts_with(&nibbles));

        // Test with same nibbles
        assert!(nibbles.starts_with(&nibbles));

        // Test with prefix
        let prefix = Nibbles::from_nibbles([1, 2]);
        assert!(nibbles.starts_with(&prefix));
        assert!(!prefix.starts_with(&nibbles));

        // Test with different first nibble
        let different = Nibbles::from_nibbles([2, 2, 3, 4]);
        assert!(!nibbles.starts_with(&different));

        // Test with longer sequence
        let longer = Nibbles::from_nibbles([1, 2, 3, 4, 5, 6]);
        assert!(!nibbles.starts_with(&longer));

        // Test with even nibbles and odd prefix
        let even_nibbles = Nibbles::from_nibbles([1, 2, 3, 4]);
        let odd_prefix = Nibbles::from_nibbles([1, 2, 3]);
        assert!(even_nibbles.starts_with(&odd_prefix));

        // Test with odd nibbles and even prefix
        let odd_nibbles = Nibbles::from_nibbles([1, 2, 3]);
        let even_prefix = Nibbles::from_nibbles([1, 2]);
        assert!(odd_nibbles.starts_with(&even_prefix));
    }

    #[test]
    fn slice() {
        // Test with empty nibbles
        let empty = Nibbles::default();
        assert_eq!(empty.slice(..), empty);

        // Test with even number of nibbles
        let even = Nibbles::from_nibbles([0, 1, 2, 3, 4, 5]);

        // Full slice
        assert_eq!(even.slice(..), even);
        assert_eq!(even.slice(0..6), even);

        // Empty slice
        assert_eq!(even.slice(3..3), Nibbles::default());

        // Beginning slices (even start)
        assert_eq!(even.slice(0..2), Nibbles::from_iter(0..2));

        // Middle slices (even start, even end)
        assert_eq!(even.slice(2..4), Nibbles::from_iter(2..4));

        // End slices (even start)
        assert_eq!(even.slice(4..6), Nibbles::from_iter(4..6));

        // Test with odd number of nibbles
        let odd = Nibbles::from_iter(0..5);

        // Full slice
        assert_eq!(odd.slice(..), odd);
        assert_eq!(odd.slice(0..5), odd);

        // Beginning slices (odd length)
        assert_eq!(odd.slice(0..3), Nibbles::from_iter(0..3));

        // Middle slices with odd start
        assert_eq!(odd.slice(1..4), Nibbles::from_iter(1..4));

        // Middle slices with odd end
        assert_eq!(odd.slice(1..3), Nibbles::from_iter(1..3));

        // End slices (odd start)
        assert_eq!(odd.slice(2..5), Nibbles::from_iter(2..5));

        // Special cases - both odd start and end
        assert_eq!(odd.slice(1..4), Nibbles::from_iter(1..4));

        // Single nibble slices
        assert_eq!(even.slice(2..3), Nibbles::from_iter(2..3));

        assert_eq!(even.slice(3..4), Nibbles::from_iter(3..4));

        // Test with alternate syntax
        assert_eq!(even.slice(2..), Nibbles::from_iter(2..6));
        assert_eq!(even.slice(..4), Nibbles::from_iter(0..4));
        assert_eq!(even.slice(..=3), Nibbles::from_iter(0..4));

        // More complex test case with the max length array sliced at the end
        assert_eq!(
            Nibbles::from_iter((0..16).cycle().take(64)).slice(1..),
            Nibbles::from_iter((0..16).cycle().take(64).skip(1))
        );
    }

    #[test]
    fn common_prefix_length() {
        // Test with empty nibbles
        let empty = Nibbles::default();
        assert_eq!(empty.common_prefix_length(&empty), 0);

        // Test with same nibbles
        let nibbles1 = Nibbles::from_nibbles([1, 2, 3, 4]);
        let nibbles2 = Nibbles::from_nibbles([1, 2, 3, 4]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 4);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 4);

        // Test with partial common prefix (byte aligned)
        let nibbles1 = Nibbles::from_nibbles([1, 2, 3, 4]);
        let nibbles2 = Nibbles::from_nibbles([1, 2, 5, 6]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 2);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 2);

        // Test with partial common prefix (half-byte aligned)
        let nibbles1 = Nibbles::from_nibbles([1, 2, 3, 4]);
        let nibbles2 = Nibbles::from_nibbles([1, 2, 3, 7]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 3);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 3);

        // Test with no common prefix
        let nibbles1 = Nibbles::from_nibbles([5, 6, 7, 8]);
        let nibbles2 = Nibbles::from_nibbles([1, 2, 3, 4]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 0);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 0);

        // Test with different lengths but common prefix
        let nibbles1 = Nibbles::from_nibbles([1, 2, 3, 4, 5, 6]);
        let nibbles2 = Nibbles::from_nibbles([1, 2, 3]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 3);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 3);

        // Test with odd number of nibbles
        let nibbles1 = Nibbles::from_nibbles([1, 2, 3]);
        let nibbles2 = Nibbles::from_nibbles([1, 2, 7]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 2);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 2);

        // Test with half-byte difference in first byte
        let nibbles1 = Nibbles::from_nibbles([1, 2, 3, 4]);
        let nibbles2 = Nibbles::from_nibbles([5, 2, 3, 4]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 0);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 0);

        // Test with one empty and one non-empty
        let nibbles1 = Nibbles::from_nibbles([1, 2, 3, 4]);
        assert_eq!(nibbles1.common_prefix_length(&empty), 0);
        assert_eq!(empty.common_prefix_length(&nibbles1), 0);

        // Test with longer sequences (16 nibbles)
        let nibbles1 =
            Nibbles::from_nibbles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]);
        let nibbles2 =
            Nibbles::from_nibbles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 15);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 15);

        // Test with different lengths but same prefix (32 vs 16 nibbles)
        let nibbles1 =
            Nibbles::from_nibbles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]);
        let nibbles2 = Nibbles::from_nibbles([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 0,
        ]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 16);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 16);

        // Test with very long sequences (32 nibbles) with different endings
        let nibbles1 = Nibbles::from_nibbles([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 0,
        ]);
        let nibbles2 = Nibbles::from_nibbles([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 1,
        ]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 31);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 31);

        // Test with 48 nibbles with different endings
        let nibbles1 = Nibbles::from_nibbles([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
        ]);
        let nibbles2 = Nibbles::from_nibbles([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1,
        ]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 47);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 47);

        // Test with 64 nibbles with different endings
        let nibbles1 = Nibbles::from_nibbles([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
        ]);
        let nibbles2 = Nibbles::from_nibbles([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1,
        ]);
        assert_eq!(nibbles1.common_prefix_length(&nibbles2), 63);
        assert_eq!(nibbles2.common_prefix_length(&nibbles1), 63);

        let current = Nibbles::from_nibbles([0u8; 64]);
        let path = Nibbles::from_nibbles([vec![0u8; 63], vec![2u8]].concat());
        assert_eq!(current.common_prefix_length(&path), 63);

        let current = Nibbles::from_nibbles([0u8; 63]);
        let path = Nibbles::from_nibbles([vec![0u8; 62], vec![1u8], vec![0u8]].concat());
        assert_eq!(current.common_prefix_length(&path), 62);
    }

    #[test]
    fn truncate() {
        // Test truncating empty nibbles
        let mut nibbles = Nibbles::default();
        nibbles.truncate(0);
        assert_eq!(nibbles, Nibbles::default());

        // Test truncating to zero length
        let mut nibbles = Nibbles::from_nibbles([1, 2, 3, 4]);
        nibbles.truncate(0);
        assert_eq!(nibbles, Nibbles::default());

        // Test truncating to same length (should be no-op)
        let mut nibbles = Nibbles::from_nibbles([1, 2, 3, 4]);
        nibbles.truncate(4);
        assert_eq!(nibbles, Nibbles::from_nibbles([1, 2, 3, 4]));

        // Individual nibble test with a simple 2-nibble truncation
        let mut nibbles = Nibbles::from_nibbles([1, 2, 3, 4]);
        nibbles.truncate(2);
        assert_eq!(nibbles, Nibbles::from_nibbles([1, 2]));

        // Test simple truncation
        let mut nibbles = Nibbles::from_nibbles([1, 2, 3, 4]);
        nibbles.truncate(2);
        assert_eq!(nibbles, Nibbles::from_nibbles([1, 2]));

        // Test truncating to single nibble
        let mut nibbles = Nibbles::from_nibbles([5, 6, 7, 8]);
        nibbles.truncate(1);
        assert_eq!(nibbles, Nibbles::from_nibbles([5]));
    }

    #[test]
    fn push_unchecked() {
        // Test pushing to empty nibbles
        let mut nibbles = Nibbles::default();
        nibbles.push_unchecked(0x5);
        assert_eq!(nibbles, Nibbles::from_nibbles([0x5]));

        // Test pushing a second nibble
        nibbles.push_unchecked(0xA);
        assert_eq!(nibbles, Nibbles::from_nibbles([0x5, 0xA]));

        // Test pushing multiple nibbles to build a sequence
        let mut nibbles = Nibbles::default();
        for nibble in [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8] {
            nibbles.push_unchecked(nibble);
        }
        assert_eq!(nibbles, Nibbles::from_nibbles([0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8]));

        // Test pushing nibbles with values that exceed 4 bits (should be masked to 0x0F)
        let mut nibbles = Nibbles::default();
        nibbles.push_unchecked(0xFF); // Should become 0xF
        nibbles.push_unchecked(0x1A); // Should become 0xA
        nibbles.push_unchecked(0x25); // Should become 0x5
        assert_eq!(nibbles, Nibbles::from_nibbles([0xF, 0xA, 0x5]));

        // Test pushing to existing nibbles (adding to the end)
        let mut nibbles = Nibbles::from_nibbles([0x1, 0x2, 0x3]);
        nibbles.push_unchecked(0x4);
        assert_eq!(nibbles, Nibbles::from_nibbles([0x1, 0x2, 0x3, 0x4]));

        // Test boundary values (0 and 15)
        let mut nibbles = Nibbles::default();
        nibbles.push_unchecked(0x0);
        nibbles.push_unchecked(0xF);
        assert_eq!(nibbles, Nibbles::from_nibbles([0x0, 0xF]));

        // Test pushing many nibbles to verify no overflow issues
        let mut nibbles = Nibbles::default();
        let test_sequence: Vec<u8> = (0..32).map(|i| i % 16).collect();
        for &nibble in &test_sequence {
            nibbles.push_unchecked(nibble);
        }
        assert_eq!(nibbles, Nibbles::from_nibbles(test_sequence));
    }

    // #[test]
    // fn unpack() {
    //     for (test_idx, bytes) in (0..=32).map(|i| vec![0xFF; i]).enumerate() {
    //         let packed_nibbles = Nibbles::unpack(&bytes);
    //         let nibbles = Nibbles::unpack(&bytes);

    //         assert_eq!(
    //             packed_nibbles.len(),
    //             nibbles.len(),
    //             "Test case {}: Length mismatch for bytes {:?}",
    //             test_idx,
    //             bytes
    //         );
    //         assert_eq!(
    //             packed_nibbles.len(),
    //             bytes.len() * 2,
    //             "Test case {}: Expected length to be 2x byte length",
    //             test_idx
    //         );

    //         // Compare each nibble individually
    //         for i in 0..packed_nibbles.len() {
    //             assert_eq!(packed_nibbles.get_nibble(i), nibbles[i],
    //                        "Test case {}: Nibble at index {} differs for bytes {:?}:
    // Nibbles={:?}, Nibbles={:?}",                        test_idx, i, bytes,
    // packed_nibbles.get_nibble(i), nibbles[i]);         }
    //     }
    // }

    #[cfg(feature = "arbitrary")]
    mod arbitrary {
        use super::*;
        use proptest::{collection::vec, prelude::*};

        proptest::proptest! {
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn pack_unpack_roundtrip(input in vec(any::<u8>(), 0..64)) {
                let nibbles = Nibbles::unpack(&input);
                prop_assert!(valid_nibbles(&nibbles));
                let packed = nibbles.pack();
                prop_assert_eq!(&packed[..], input);
            }
        }
    }
}
