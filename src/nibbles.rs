use core::{
    cmp::{self, Ordering},
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

/// The size of [`U256`] in nibbles.
const NIBBLES: usize = 64;

/// This array contains 65 bitmasks used in [`Nibbles::slice`].
///
/// Each mask is a [`U256`] where:
/// - Index 0 is just 0 (no bits set)
/// - Index 1 has the highest 4 bits set (one nibble)
/// - Index 2 has the highest 8 bits set (two nibbles)
/// - ...and so on
/// - Index 64 has all bits set ([`U256::MAX`])
static SLICE_MASKS: [U256; 65] = {
    let mut masks = [U256::ZERO; 65];
    let mut i = 0;
    while i <= NIBBLES {
        masks[i] = if i == 0 { U256::ZERO } else { U256::MAX.wrapping_shl((NIBBLES - i) * 4) };
        i += 1;
    }
    masks
};

/// This array contains 65 increment masks used in [`Nibbles::increment`].
///
/// Each mask is a [`U256`] equal to `1 << ((64 - i) * 4)`.
static INCREMENT_MASKS: [U256; 65] = {
    let mut masks = [U256::ZERO; 65];
    let mut i = 0;
    while i <= NIBBLES {
        masks[i] = U256::ONE.wrapping_shl((NIBBLES - i) * 4);
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
/// The internal representation is currently a [`U256`] that stores two nibbles per byte. Nibbles
/// are stored inline (on the stack), and can be up to a length of 64 nibbles, or 32 unpacked bytes.
///
/// Nibbles are stored with most significant bits set first, meaning that a nibble sequence `0x101`
/// will be stored as `0x101...0`, and not `0x0...101`.
///
/// # Examples
///
/// ```
/// use nybbles::Nibbles;
///
/// let bytes = [0xAB, 0xCD];
/// let nibbles = Nibbles::unpack(&bytes);
/// assert_eq!(nibbles, Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D]));
/// assert_eq!(nibbles.to_vec(), vec![0x0A, 0x0B, 0x0C, 0x0D]);
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
    /// The nibbles themselves, stored as a 256-bit unsigned integer with most significant bits set
    /// first.
    pub(crate) nibbles: U256,
}

impl fmt::Debug for Nibbles {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "Nibbles(0x)")
        } else {
            let shifted = self.nibbles.wrapping_shr((NIBBLES - self.len()) * 4);
            write!(f, "Nibbles(0x{:0width$x})", shifted, width = self.len())
        }
    }
}

// Deriving [`Ord`] for [`Nibbles`] is not correct, because they will be compared as unsigned
// integers without accounting for length. This is incorrect, because `0x1` should be considered
// greater than `0x02`.
impl Ord for Nibbles {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let self_len = self.len().div_ceil(2);
        let other_len = other.len().div_ceil(2);
        let l = cmp::min(self_len, other_len);

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &self.nibbles.as_le_slice()[U256::BYTES - l..];
        let rhs = &other.nibbles.as_le_slice()[U256::BYTES - l..];

        for i in (0..l).rev() {
            match lhs[i].cmp(&rhs[i]) {
                Ordering::Equal => (),
                non_eq => return non_eq,
            }
        }

        self.len().cmp(&other.len())
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
        /// List of possible nibbles to return static references. It's a hack that allows us to
        /// return a reference to a nibble, even though we cannot address nibbles directly and must
        /// go through bytes first.
        static NIBBLES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        &NIBBLES[self.get_unchecked(index) as usize]
    }
}

impl FromIterator<u8> for Nibbles {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let mut nibbles = Self::default();
        for n in iter {
            nibbles.push(n);
        }
        nibbles
    }
}

#[cfg(feature = "rlp")]
impl alloy_rlp::Encodable for Nibbles {
    #[inline]
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        alloy_rlp::Header { list: true, payload_length: self.len() }.encode(out);
        for i in 0..self.len() {
            self.get_unchecked(i).encode(out);
        }
    }

    #[inline]
    fn length(&self) -> usize {
        let payload_length = self.length as usize;
        payload_length + alloy_rlp::length_of_length(payload_length)
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

    /// Same as [`FromIterator`] implementation, but skips the validity check.
    pub fn from_iter_unchecked<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        let mut packed = Self::default();
        for n in iter {
            packed.push_unchecked(n);
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
    /// assert_eq!(nibbles.to_vec(), vec![0x0A, 0x0B, 0x0C, 0x0D]);
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
    /// assert_eq!(nibbles.to_vec(), vec![0x0A, 0x0B, 0x0C, 0x0D]);
    ///
    /// // Invalid value!
    /// let nibbles = Nibbles::from_nibbles_unchecked(&[0xFF]);
    /// assert_eq!(nibbles.to_vec(), vec![0x0F]);
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
    /// assert_eq!(nibbles.to_vec(), vec![0x0A, 0x0B, 0x0C, 0x0D]);
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
    /// assert_eq!(nibbles.to_vec(), vec![0x0A, 0x0B, 0x0C, 0x0D]);
    ///
    /// // Invalid value!
    /// let nibbles = Nibbles::from_vec_unchecked(vec![0xFF]);
    /// assert_eq!(nibbles.to_vec(), vec![0x0F]);
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
    /// Panics if the length of the input is greater than 32 bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::unpack(&[0xAB, 0xCD]);
    /// assert_eq!(nibbles.to_vec(), vec![0x0A, 0x0B, 0x0C, 0x0D]);
    /// ```
    #[inline]
    pub fn unpack<T: AsRef<[u8]>>(data: T) -> Self {
        assert!(data.as_ref().len() <= U256::BYTES);
        // SAFETY: we checked that the length is less than or equal to the size of U256
        unsafe { Self::unpack_unchecked(data) }
    }

    /// Converts a byte slice into a [`Nibbles`] instance containing the nibbles (half-bytes or 4
    /// bits) that make up the input byte data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the length of the input is less than or equal to the size of
    /// U256, which is 32 bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// // SAFETY: the length of the input is less than 32 bytes.
    /// let nibbles = unsafe { Nibbles::unpack_unchecked(&[0xAB, 0xCD]) };
    /// assert_eq!(nibbles.to_vec(), vec![0x0A, 0x0B, 0x0C, 0x0D]);
    /// ```
    pub unsafe fn unpack_unchecked<T: AsRef<[u8]>>(data: T) -> Self {
        let data = data.as_ref();
        let length = (data.len() * 2) as u8;
        debug_assert!(length as usize <= NIBBLES);

        let mut nibbles = U256::ZERO;

        // Source pointer is at the beginning
        let mut src = data.as_ptr().cast::<u8>();
        // Move destination pointer to the end of the little endian slice
        let mut dst = nibbles.as_le_slice_mut().as_mut_ptr().add(U256::BYTES);
        // On each iteration, decrement the destination pointer by one, set the destination
        // byte, and increment the source pointer by one
        for _ in 0..data.len() {
            dst = dst.sub(1);
            *dst = *src;
            src = src.add(1);
        }

        Self { length, nibbles }
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
        let packed_len = self.len().div_ceil(2);
        unsafe { smallvec_with(packed_len, |out| self.pack_to_slice_unchecked(out)) }
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
        assert!(out.len() >= self.len().div_ceil(2));
        // SAFETY: asserted length.
        unsafe { self.pack_to_unchecked(out.as_mut_ptr()) }
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
    pub unsafe fn pack_to_unchecked(&self, ptr: *mut u8) {
        let slice = slice::from_raw_parts_mut(ptr.cast(), self.len().div_ceil(2));
        pack_to_unchecked(self, slice);
    }

    /// Packs the nibbles into the given slice without checking its length.
    ///
    /// # Safety
    ///
    /// `out` must be valid for at least `(self.len() + 1) / 2` bytes.
    #[inline]
    pub unsafe fn pack_to_slice_unchecked(&self, out: &mut [MaybeUninit<u8>]) {
        pack_to_unchecked(self, out)
    }

    /// Converts the nibbles into a vector of nibbles.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut nibbles = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            nibbles.push(self.get_unchecked(i));
        }
        nibbles
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
        if likely((i < usize::MAX) & self.check_index(i.wrapping_add(1))) {
            Some(self.get_byte_unchecked(i))
        } else {
            None
        }
    }

    /// Gets the byte at the given index by combining two consecutive nibbles.
    ///
    /// # Panics
    ///
    /// Panics if `i..i + 1` is out of bounds.
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
    pub fn get_byte_unchecked(&self, i: usize) -> u8 {
        self.assert_index(i);
        if i % 2 == 0 {
            self.nibbles.as_le_slice()[U256::BYTES - i / 2 - 1]
        } else {
            self.get_unchecked(i) << 4 | self.get_unchecked(i + 1)
        }
    }

    /// Increments the nibble sequence by one.
    #[inline]
    pub fn increment(&self) -> Option<Self> {
        if self.nibbles == SLICE_MASKS[self.len()] {
            return None;
        }

        let mut incremented = *self;
        let add = INCREMENT_MASKS[self.len()];
        incremented.nibbles += add;
        Some(incremented)
    }

    /// The last element of the hex vector is used to determine whether the nibble sequence
    /// represents a leaf or an extension node. If the last element is 0x10 (16), then it's a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.last() == Some(16)
    }

    /// Returns `true` if this nibble sequence starts with the given prefix.
    pub fn starts_with(&self, other: &Self) -> bool {
        // Fast path: if lengths don't allow prefix, return false
        if other.len() > self.len() {
            return false;
        }

        // Fast path: empty prefix always matches
        if other.is_empty() {
            return true;
        }

        // Direct comparison using masks
        let mask = SLICE_MASKS[other.len()];
        (self.nibbles & mask) == other.nibbles
    }

    /// Returns `true` if this nibble sequence ends with the given prefix.
    pub fn ends_with(&self, other: &Self) -> bool {
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
            if self.get_unchecked(self.len() - i - 1) != other.get_unchecked(other.len() - i - 1) {
                return false;
            }
            i += 1;
        }

        true
    }

    /// Returns the nibble at the given index.
    pub fn get(&self, i: usize) -> Option<u8> {
        if self.check_index(i) {
            Some(self.get_unchecked(i))
        } else {
            None
        }
    }

    /// Returns the nibble at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    #[track_caller]
    pub fn get_unchecked(&self, i: usize) -> u8 {
        self.assert_index(i);
        let byte = self.nibbles.as_le_slice()[U256::BYTES - i / 2 - 1];
        if i % 2 == 0 {
            byte >> 4
        } else {
            byte & 0x0F
        }
    }

    /// Sets the nibble at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds, or if `value` is not a valid nibble (`0..=0x0f`).
    #[inline]
    #[track_caller]
    pub fn set_at(&mut self, i: usize, value: u8) {
        assert!(self.check_index(i) && value <= 0xf);
        // SAFETY: index is checked above
        unsafe { self.set_at_unchecked(i, value) };
    }

    /// Sets the nibble at the given index, without checking its validity.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    #[inline]
    pub unsafe fn set_at_unchecked(&mut self, i: usize, value: u8) {
        let byte_index = U256::BYTES - i / 2 - 1;
        // SAFETY: index checked above
        let byte = unsafe { &mut self.nibbles.as_le_slice_mut()[byte_index] };
        if i % 2 == 0 {
            *byte = *byte & 0x0f | value << 4;
        } else {
            *byte = *byte & 0xf0 | value;
        }
    }

    /// Returns the first nibble of this nibble sequence.
    pub fn first(&self) -> Option<u8> {
        self.get(0)
    }

    /// Returns the last nibble of this nibble sequence.
    pub fn last(&self) -> Option<u8> {
        let len = self.len();
        if len == 0 {
            None
        } else {
            Some(self.get_unchecked(len - 1))
        }
    }

    /// Returns the length of the common prefix between this nibble sequence and the given.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let a = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
    /// let b = Nibbles::from_nibbles(&[0x0A, 0x0B, 0x0C, 0x0E]);
    /// assert_eq!(a.common_prefix_length(&b), 3);
    /// ```
    pub fn common_prefix_length(&self, other: &Self) -> usize {
        // Handle empty cases
        if self.is_empty() || other.is_empty() {
            return 0;
        }

        let min_nibble_len = self.len().min(other.len());

        // Fast path for small sequences that fit in one u64 limb
        if min_nibble_len <= 16 {
            // Extract the highest u64 limb which contains all the nibbles
            let self_limb = self.nibbles.as_limbs()[3];
            let other_limb = other.nibbles.as_limbs()[3];

            // Create mask for the nibbles we care about
            let mask = u64::MAX << ((16 - min_nibble_len) * 4);
            let xor = (self_limb & mask) ^ (other_limb & mask);

            if xor == 0 {
                return min_nibble_len;
            } else {
                return xor.leading_zeros() as usize / 4;
            }
        }

        let xor = if min_nibble_len == NIBBLES && self.len() == other.len() {
            // No need to mask for 64 nibble sequences, just XOR
            self.nibbles ^ other.nibbles
        } else {
            // For other lengths, mask the nibbles we care about, and then XOR
            let mask = SLICE_MASKS[min_nibble_len];
            let masked_self = self.nibbles & mask;
            let masked_other = other.nibbles & mask;
            masked_self ^ masked_other
        };

        if xor == U256::ZERO {
            min_nibble_len
        } else {
            xor.leading_zeros() / 4
        }
    }

    /// Returns the total number of bits in this [`Nibbles`].
    #[inline]
    const fn bit_len(&self) -> usize {
        self.length as usize * 4
    }

    /// Returns `true` if this [`Nibbles`] is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the total number of nibbles in this [`Nibbles`].
    #[inline]
    pub const fn len(&self) -> usize {
        let len = self.length as usize;
        unsafe { core::hint::assert_unchecked(len <= 64) };
        len
    }

    /// Returns a mutable reference to the underlying [`U256`].
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    #[inline]
    pub fn as_mut_uint_unchecked(&mut self) -> &mut U256 {
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

        // Optimize for common case where start == 0
        let nibbles = if start == 0 {
            // When slicing from the beginning, we only need the end mask
            // This avoids the XOR operation
            self.nibbles.bitand(SLICE_MASKS[end])
        } else {
            // For middle and to_end cases, always shift first
            let shifted = self.nibbles.wrapping_shl(start * 4);
            if slice_to_end {
                // When slicing to the end, no mask needed after shift
                shifted
            } else {
                // For middle slices, apply end mask after shift
                shifted.bitand(SLICE_MASKS[end - start])
            }
        };

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

    /// Join two nibble sequences together.
    #[inline]
    pub const fn join(&self, other: &Self) -> Self {
        let mut new = *self;
        if other.is_empty() {
            return new;
        }

        new.nibbles = new.nibbles.bitor(other.nibbles.wrapping_shr(self.bit_len()));
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
    /// Note that only the low nibble of the byte is used. For example, for byte `0x12`, only the
    /// nibble `0x2` is pushed.
    #[inline]
    pub fn push_unchecked(&mut self, nibble: u8) {
        let nibble_val = (nibble & 0x0F) as u64;
        if nibble_val == 0 {
            // Nothing to do, limb nibbles are already set to zero by default
            self.length += 1;
            return;
        }

        let bit_pos = (NIBBLES - self.length as usize - 1) * 4;
        let limb_idx = bit_pos / NIBBLES;
        let shift_in_limb = bit_pos % NIBBLES;

        // SAFETY: limb_idx is always valid because bit_pos < 256
        unsafe {
            let limbs = self.nibbles.as_limbs_mut();
            limbs[limb_idx] |= nibble_val << shift_in_limb;
        }

        self.length += 1;
    }

    /// Pops a nibble from the end of the current nibbles.
    pub fn pop(&mut self) -> Option<u8> {
        if self.length == 0 {
            return None;
        }

        // The last nibble is at bit position (64 - length) * 4 from the MSB
        let shift = (NIBBLES - self.length as usize) * 4;

        // Extract the nibble - after shifting right, it's in the lowest bits of limb 0
        let nibble = ((self.nibbles.wrapping_shr(shift).as_limbs()[0]) & 0xF) as u8;

        // Clear the nibble using a more efficient mask creation
        // Instead of U256::from(0xF_u8) << shift, we can create the mask directly
        let mask_limb_idx = shift / 64;
        let mask_shift = shift % 64;

        if mask_limb_idx < 4 {
            // SAFETY: We know the limb index is valid
            unsafe {
                let limbs = self.nibbles.as_limbs_mut();
                limbs[mask_limb_idx] &= !(0xF << mask_shift);
            }
        }

        self.length -= 1;
        Some(nibble)
    }

    /// Extend the current nibbles with another nibbles.
    pub fn extend(&mut self, other: &Nibbles) {
        if other.is_empty() {
            return;
        }

        self.nibbles |= other.nibbles.wrapping_shr(self.bit_len());
        self.length += other.length;
    }

    /// Extend the current nibbles with another byte slice.
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    pub fn extend_from_slice(&mut self, other: &[u8]) {
        if other.is_empty() {
            return;
        }

        let len_bytes = other.len();
        let mut other = U256::from_be_slice(other);
        if len_bytes > 0 {
            other = other.wrapping_shl((U256::BYTES - len_bytes) * 8);
        }
        self.nibbles |= other.wrapping_shr(self.bit_len());
        self.length += (len_bytes * 2) as u8;
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

    #[inline]
    fn check_index(&self, i: usize) -> bool {
        i < self.len()
    }

    #[inline]
    fn assert_index(&self, i: usize) {
        let len = self.len();
        if i >= len {
            panic_invalid_index(len, i);
        }
    }
}

/// Packs the nibbles into the given slice without checking its length.
///
/// # Safety
///
/// `out` must be valid for at least `(self.len() + 1) / 2` bytes.
#[inline]
unsafe fn pack_to_unchecked(nibbles: &Nibbles, out: &mut [MaybeUninit<u8>]) {
    let byte_len = nibbles.len().div_ceil(2);
    debug_assert!(out.len() >= byte_len);
    // Move source pointer to the end of the little endian slice
    let mut src = nibbles.nibbles.as_le_slice().as_ptr().add(U256::BYTES);
    // Destination pointer is at the beginning of the output slice
    let mut dst = out.as_mut_ptr().cast::<u8>();
    // On each iteration, decrement the source pointer by one, set the destination byte, and
    // increment the destination pointer by one
    for _ in 0..byte_len {
        src = src.sub(1);
        *dst = *src;
        dst = dst.add(1);
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
#[inline(never)]
#[cfg_attr(debug_assertions, track_caller)]
const fn panic_invalid_nibbles() -> ! {
    panic!("attempted to create invalid nibbles");
}

#[cold]
#[inline(never)]
#[cfg_attr(debug_assertions, track_caller)]
fn panic_invalid_index(len: usize, i: usize) -> ! {
    panic!("index out of bounds: {i} for nibbles of length {len}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack() {
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
                "input: {input:x?}, expected: {expected:x?}, got: {encoded:x?}",
            );
        }
    }

    #[test]
    fn get_unchecked() {
        for len in 0..NIBBLES {
            let raw = (0..16).cycle().take(len).collect::<Vec<u8>>();
            let nibbles = Nibbles::from_nibbles(&raw);
            for (i, raw_nibble) in raw.into_iter().enumerate() {
                assert_eq!(nibbles.get_unchecked(i), raw_nibble);
            }
        }
    }

    #[test]
    fn set_at_unchecked() {
        for len in 0..=NIBBLES {
            let raw = (0..16).cycle().take(len).collect::<Vec<u8>>();
            let mut nibbles = Nibbles::from_nibbles(&raw);
            for (i, raw_nibble) in raw.iter().enumerate() {
                let new_nibble = (raw_nibble + 1) % 16;
                unsafe { nibbles.set_at_unchecked(i, new_nibble) };

                let mut new_raw_nibbles = nibbles.clone().to_vec();
                new_raw_nibbles[i] = new_nibble;
                let new_nibbles = Nibbles::from_nibbles(&new_raw_nibbles);
                assert_eq!(nibbles, new_nibbles,);
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
    fn get_byte() {
        let nibbles = Nibbles::from_nibbles([0x0A, 0x0B, 0x0C, 0x0D]);
        assert_eq!(nibbles.get_byte(0), Some(0xAB));
        assert_eq!(nibbles.get_byte(1), Some(0xBC));
        assert_eq!(nibbles.get_byte(2), Some(0xCD));
        assert_eq!(nibbles.get_byte(3), None);
        assert_eq!(nibbles.get_byte(usize::MAX), None);
    }

    #[test]
    fn get_byte_unchecked() {
        let nibbles = Nibbles::from_nibbles([0x0A, 0x0B, 0x0C, 0x0D]);
        assert_eq!(nibbles.get_byte_unchecked(0), 0xAB);
        assert_eq!(nibbles.get_byte_unchecked(1), 0xBC);
        assert_eq!(nibbles.get_byte_unchecked(2), 0xCD);
    }

    #[test]
    fn clone() {
        let a = Nibbles::from_nibbles([1, 2, 3]);
        #[allow(clippy::redundant_clone)]
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn ord() {
        // Test empty nibbles
        let nibbles1 = Nibbles::default();
        let nibbles2 = Nibbles::default();
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Equal);

        // Test with one empty
        let nibbles1 = Nibbles::default();
        let nibbles2 = Nibbles::from_nibbles([0]);
        assert_eq!(nibbles2.cmp(&nibbles1), Ordering::Greater);

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
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Less);

        let nibbles1 = Nibbles::from_nibbles([0x1]);
        let nibbles2 = Nibbles::from_nibbles([0x0, 0x2]);
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Greater);

        let nibbles1 = Nibbles::from_nibbles([vec![0; 61], vec![1; 1], vec![0; 1]].concat());
        let nibbles2 = Nibbles::from_nibbles([vec![0; 61], vec![1; 1], vec![0; 2]].concat());
        assert_eq!(nibbles1.cmp(&nibbles2), Ordering::Less);
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

    #[test]
    fn unpack() {
        for (test_idx, bytes) in (0..=32).map(|i| vec![0xFF; i]).enumerate() {
            let packed_nibbles = Nibbles::unpack(&bytes);
            let nibbles = Nibbles::unpack(&bytes);

            assert_eq!(
                packed_nibbles.len(),
                nibbles.len(),
                "Test case {test_idx}: Length mismatch for bytes {bytes:?}",
            );
            assert_eq!(
                packed_nibbles.len(),
                bytes.len() * 2,
                "Test case {test_idx}: Expected length to be 2x byte length",
            );

            // Compare each nibble individually
            for i in 0..packed_nibbles.len() {
                assert_eq!(
                    packed_nibbles.get_unchecked(i),
                    nibbles[i],
                    "Test case {}: Nibble at index {} differs for bytes {:?}:
    Nibbles={:?}, Nibbles={:?}",
                    test_idx,
                    i,
                    bytes,
                    packed_nibbles.get_unchecked(i),
                    nibbles[i]
                );
            }
        }

        let nibbles = Nibbles::unpack([0xAB, 0xCD]);
        assert_eq!(nibbles.to_vec(), vec![0x0A, 0x0B, 0x0C, 0x0D]);
    }

    #[test]
    fn increment() {
        // Test basic increment
        assert_eq!(
            Nibbles::from_nibbles([0x0, 0x0, 0x0]).increment().unwrap(),
            Nibbles::from_nibbles([0x0, 0x0, 0x1])
        );

        // Test increment with carry
        assert_eq!(
            Nibbles::from_nibbles([0x0, 0x0, 0xF]).increment().unwrap(),
            Nibbles::from_nibbles([0x0, 0x1, 0x0])
        );

        // Test multiple carries
        assert_eq!(
            Nibbles::from_nibbles([0x0, 0xF, 0xF]).increment().unwrap(),
            Nibbles::from_nibbles([0x1, 0x0, 0x0])
        );

        // Test increment from all F's except first nibble
        assert_eq!(
            Nibbles::from_nibbles([0xE, 0xF, 0xF]).increment().unwrap(),
            Nibbles::from_nibbles([0xF, 0x0, 0x0])
        );

        // Test overflow - all nibbles are 0xF
        assert_eq!(Nibbles::from_nibbles([0xF, 0xF, 0xF]).increment(), None);

        // Test empty nibbles
        assert_eq!(Nibbles::new().increment(), None);

        // Test single nibble
        assert_eq!(Nibbles::from_nibbles([0x5]).increment().unwrap(), Nibbles::from_nibbles([0x6]));

        // Test single nibble at max
        assert_eq!(Nibbles::from_nibbles([0xF]).increment(), None);

        // Test longer sequence
        assert_eq!(
            Nibbles::from_nibbles([0x1, 0x2, 0x3, 0x4, 0x5]).increment().unwrap(),
            Nibbles::from_nibbles([0x1, 0x2, 0x3, 0x4, 0x6])
        );

        // Test longer sequence with carries
        assert_eq!(
            Nibbles::from_nibbles([0x1, 0x2, 0x3, 0xF, 0xF]).increment().unwrap(),
            Nibbles::from_nibbles([0x1, 0x2, 0x4, 0x0, 0x0])
        );
    }

    #[cfg(feature = "arbitrary")]
    mod arbitrary {
        use super::*;
        use proptest::{collection::vec, prelude::*};

        proptest::proptest! {
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn pack_unpack_roundtrip(input in vec(any::<u8>(), 0..32)) {
                let nibbles = Nibbles::unpack(&input);
                prop_assert!(valid_nibbles(&nibbles.to_vec()));
                let packed = nibbles.pack();
                prop_assert_eq!(&packed[..], input);
            }
        }
    }
}
