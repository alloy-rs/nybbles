use core::{borrow::Borrow, fmt, mem::MaybeUninit, ops::Index, slice};
use smallvec::SmallVec;

#[cfg(not(feature = "nightly"))]
#[allow(unused_imports)]
use core::convert::{identity as likely, identity as unlikely};
#[cfg(feature = "nightly")]
#[allow(unused_imports)]
use core::intrinsics::{likely, unlikely};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

type Repr = SmallVec<[u8; 64]>;

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
#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct Nibbles(Repr);

impl core::ops::Deref for Nibbles {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

// Override `SmallVec::from` in the default `Clone` implementation since it's not specialized for
// `Copy` types.
impl Clone for Nibbles {
    #[inline]
    fn clone(&self) -> Self {
        Self(SmallVec::from_slice(&self.0))
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0);
    }
}

impl fmt::Debug for Nibbles {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Nibbles(0x{})", const_hex::encode(self.as_slice()))
    }
}

impl From<Nibbles> for Vec<u8> {
    #[inline]
    fn from(value: Nibbles) -> Self {
        value.0.into_vec()
    }
}

impl PartialEq<[u8]> for Nibbles {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl PartialEq<Nibbles> for [u8] {
    #[inline]
    fn eq(&self, other: &Nibbles) -> bool {
        self == other.as_slice()
    }
}

impl PartialOrd<[u8]> for Nibbles {
    #[inline]
    fn partial_cmp(&self, other: &[u8]) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other)
    }
}

impl PartialOrd<Nibbles> for [u8] {
    #[inline]
    fn partial_cmp(&self, other: &Nibbles) -> Option<core::cmp::Ordering> {
        self.partial_cmp(other.as_slice())
    }
}

impl Borrow<[u8]> for Nibbles {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<Idx> core::ops::Index<Idx> for Nibbles
where
    Repr: core::ops::Index<Idx>,
{
    type Output = <Repr as core::ops::Index<Idx>>::Output;

    #[inline]
    fn index(&self, index: Idx) -> &Self::Output {
        self.0.index(index)
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
        Self(SmallVec::new_const())
    }

    /// Creates a new [`Nibbles`] instance with the given capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nybbles::Nibbles;
    /// let nibbles = Nibbles::with_capacity(10);
    /// assert_eq!(nibbles.len(), 0);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(SmallVec::with_capacity(capacity))
    }

    /// Creates a new [`Nibbles`] instance with the given nibbles.
    #[inline]
    pub fn from_repr(nibbles: Repr) -> Self {
        check_nibbles(&nibbles);
        Self::from_repr_unchecked(nibbles)
    }

    /// Creates a new [`Nibbles`] instance with the given nibbles.
    ///
    /// This will not unpack the bytes into nibbles, and will instead store the bytes as-is.
    ///
    /// Note that it is possible to create a [`Nibbles`] instance with invalid nibble values (i.e.
    /// values greater than 0xf) using this method. See [the type docs](Self) for more details.
    ///
    /// # Panics
    ///
    /// Panics if the any of the bytes is not a valid nibble (`0..=0x0f`).
    #[inline]
    pub const fn from_repr_unchecked(small_vec: Repr) -> Self {
        Self(small_vec)
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
        let nibbles = nibbles.as_ref();
        check_nibbles(nibbles);
        Self::from_nibbles_unchecked(nibbles)
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
        Self(SmallVec::from_slice(nibbles.as_ref()))
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
        Self(SmallVec::from_vec(vec))
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
        Self::unpack_(data.as_ref())
    }

    #[inline]
    fn unpack_(data: &[u8]) -> Self {
        let unpacked_len =
            data.len().checked_mul(2).expect("trying to unpack usize::MAX / 2 bytes");
        Self(unsafe { smallvec_with(unpacked_len, |out| Self::unpack_to_unchecked(data, out)) })
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
        let packed_len = self.len().div_ceil(2);
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
        self.pack_to_unchecked2(slice::from_raw_parts_mut(ptr.cast(), self.len().div_ceil(2)));
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
    #[inline]
    pub fn get_byte(&self, i: usize) -> Option<u8> {
        get_byte(self, i)
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
    #[inline]
    pub unsafe fn get_byte_unchecked(&self, i: usize) -> u8 {
        get_byte_unchecked(self, i)
    }

    /// Increments the nibble sequence by one.
    #[inline]
    pub fn increment(&self) -> Option<Self> {
        let mut incremented = self.clone();

        for nibble in incremented.0.iter_mut().rev() {
            debug_assert!(*nibble <= 0xf);
            if *nibble < 0xf {
                *nibble += 1;
                return Some(incremented);
            } else {
                *nibble = 0;
            }
        }

        None
    }

    /// The last element of the hex vector is used to determine whether the nibble sequence
    /// represents a leaf or an extension node. If the last element is 0x10 (16), then it's a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.last() == Some(16)
    }

    /// Returns `true` if this nibble sequence starts with the given prefix.
    #[inline]
    pub fn has_prefix(&self, other: &[u8]) -> bool {
        self.starts_with(other)
    }

    /// Returns the nibble at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    #[track_caller]
    pub fn at(&self, i: usize) -> usize {
        self[i] as usize
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
        self.0[i] = value;
    }

    /// Returns the first nibble of this nibble sequence.
    #[inline]
    pub fn first(&self) -> Option<u8> {
        self.0.first().copied()
    }

    /// Returns the last nibble of this nibble sequence.
    #[inline]
    pub fn last(&self) -> Option<u8> {
        self.0.last().copied()
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
    #[inline]
    pub fn common_prefix_length(&self, other: &[u8]) -> usize {
        common_prefix_length(self, other)
    }

    /// Returns a reference to the underlying byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    /// Returns a mutable reference to the underlying byte slice.
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    #[inline]
    pub fn as_mut_slice_unchecked(&mut self) -> &mut [u8] {
        &mut self.0
    }

    /// Returns a mutable reference to the underlying byte vector.
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    #[inline]
    pub fn as_mut_vec_unchecked(&mut self) -> &mut Repr {
        &mut self.0
    }

    /// Slice the current nibbles within the provided index range.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    #[inline]
    #[track_caller]
    pub fn slice<I>(&self, range: I) -> Self
    where
        Self: Index<I, Output = [u8]>,
    {
        Self::from_nibbles_unchecked(&self[range])
    }

    /// Join two nibbles together.
    #[inline]
    pub fn join(&self, b: &Self) -> Self {
        let mut nibbles = SmallVec::with_capacity(self.len() + b.len());
        nibbles.extend_from_slice(self);
        nibbles.extend_from_slice(b);
        Self(nibbles)
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
    #[inline]
    pub fn push_unchecked(&mut self, nibble: u8) {
        self.0.push(nibble);
    }

    /// Pops a nibble from the end of the current nibbles.
    #[inline]
    pub fn pop(&mut self) -> Option<u8> {
        self.0.pop()
    }

    /// Extend the current nibbles with another nibbles.
    #[inline]
    pub fn extend_from_slice(&mut self, b: &Nibbles) {
        self.0.extend_from_slice(b.as_slice());
    }

    /// Extend the current nibbles with another byte slice.
    ///
    /// Note that it is possible to create invalid [`Nibbles`] instances using this method. See
    /// [the type docs](Self) for more details.
    #[inline]
    pub fn extend_from_slice_unchecked(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }

    /// Truncates the current nibbles to the given length.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        self.0.truncate(new_len);
    }

    /// Clears the current nibbles.
    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

/// Gets the byte at the given index by combining two consecutive nibbles.
///
/// # Examples
///
/// ```
/// # use nybbles::get_byte;
/// let nibbles: &[u8] = &[0x0A, 0x0B, 0x0C, 0x0D];
/// assert_eq!(get_byte(nibbles, 0), Some(0xAB));
/// assert_eq!(get_byte(nibbles, 1), Some(0xBC));
/// assert_eq!(get_byte(nibbles, 2), Some(0xCD));
/// assert_eq!(get_byte(nibbles, 3), None);
/// ```
#[inline]
pub fn get_byte(nibbles: &[u8], i: usize) -> Option<u8> {
    if likely((i < usize::MAX) & (i.wrapping_add(1) < nibbles.len())) {
        Some(unsafe { get_byte_unchecked(nibbles, i) })
    } else {
        None
    }
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
/// # use nybbles::get_byte_unchecked;
/// let nibbles: &[u8] = &[0x0A, 0x0B, 0x0C, 0x0D];
/// // SAFETY: in range.
/// unsafe {
///     assert_eq!(get_byte_unchecked(nibbles, 0), 0xAB);
///     assert_eq!(get_byte_unchecked(nibbles, 1), 0xBC);
///     assert_eq!(get_byte_unchecked(nibbles, 2), 0xCD);
/// }
/// ```
#[inline]
pub unsafe fn get_byte_unchecked(nibbles: &[u8], i: usize) -> u8 {
    debug_assert!(
        i < usize::MAX && i + 1 < nibbles.len(),
        "index {i}..{} out of bounds of {}",
        i + 1,
        nibbles.len()
    );
    let hi = *nibbles.get_unchecked(i);
    let lo = *nibbles.get_unchecked(i + 1);
    (hi << 4) | lo
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
pub fn pack_to(nibbles: &[u8], out: &mut [u8]) {
    assert!(out.len() >= nibbles.len().div_ceil(2));
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
pub unsafe fn pack_to_unchecked(nibbles: &[u8], out: &mut [MaybeUninit<u8>]) {
    let len = nibbles.len();
    debug_assert!(out.len() >= len.div_ceil(2));
    let ptr = out.as_mut_ptr().cast::<u8>();
    for i in 0..len / 2 {
        ptr.add(i).write(get_byte_unchecked(nibbles, i * 2));
    }
    if len % 2 != 0 {
        let i = len / 2;
        ptr.add(i).write(nibbles.last().unwrap_unchecked() << 4);
    }
}

/// Returns the length of the common prefix between the two slices.
///
/// # Examples
///
/// ```
/// # use nybbles::common_prefix_length;
/// let a = &[0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F];
/// let b = &[0x0A, 0x0B, 0x0C, 0x0E];
/// assert_eq!(common_prefix_length(a, b), 3);
/// ```
#[inline]
pub fn common_prefix_length(a: &[u8], b: &[u8]) -> usize {
    let len = core::cmp::min(a.len(), b.len());
    let a = &a[..len];
    let b = &b[..len];
    for i in 0..len {
        if a[i] != b[i] {
            return i;
        }
    }
    len
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
            assert_eq!(&encoded[..], expected);
        }
    }

    #[test]
    fn slice() {
        const RAW: &[u8] = &hex!("05010406040a040203030f010805020b050c04070003070e0909070f010b0a0805020301070c0a0902040b0f000f0006040a04050f020b090701000a0a040b");

        #[track_caller]
        fn test_slice<I>(range: I, expected: &[u8])
        where
            Nibbles: Index<I, Output = [u8]>,
        {
            let nibbles = Nibbles::from_nibbles_unchecked(RAW);
            let sliced = nibbles.slice(range);
            assert_eq!(sliced, Nibbles::from_nibbles(expected));
            assert_eq!(sliced.as_slice(), expected);
        }

        test_slice(0..0, &[]);
        test_slice(0..1, &[0x05]);
        test_slice(1..1, &[]);
        test_slice(1..=1, &[0x01]);
        test_slice(0..=1, &[0x05, 0x01]);
        test_slice(0..2, &[0x05, 0x01]);

        test_slice(..0, &[]);
        test_slice(..1, &[0x05]);
        test_slice(..=1, &[0x05, 0x01]);
        test_slice(..2, &[0x05, 0x01]);

        test_slice(.., RAW);
        test_slice(..RAW.len(), RAW);
        test_slice(0.., RAW);
        test_slice(0..RAW.len(), RAW);
    }

    #[test]
    fn indexing() {
        let mut nibbles = Nibbles::from_nibbles([0x0A]);
        assert_eq!(nibbles.at(0), 0x0A);
        nibbles.set_at(0, 0x0B);
        assert_eq!(nibbles.at(0), 0x0B);
    }

    #[test]
    fn push_pop() {
        let mut nibbles = Nibbles::new();
        nibbles.push(0x0A);
        assert_eq!(nibbles[0], 0x0A);
        assert_eq!(nibbles.len(), 1);

        assert_eq!(nibbles.pop(), Some(0x0A));
        assert_eq!(nibbles.len(), 0);
    }

    #[test]
    fn get_byte_max() {
        let nibbles = Nibbles::from_nibbles([0x0A, 0x0B, 0x0C, 0x0D]);
        assert_eq!(nibbles.get_byte(usize::MAX), None);
    }

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

            /// Test that `new()` always creates empty Nibbles
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn new_always_empty(_dummy in any::<u8>()) {
                let nibbles = Nibbles::new();
                prop_assert_eq!(nibbles.len(), 0);
                prop_assert!(nibbles.is_empty());
                prop_assert_eq!(nibbles.as_slice(), &[]);
            }

            /// Test `from_nibbles()` with valid nibbles (0x0..=0xF)
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn from_nibbles_valid(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                prop_assert_eq!(nibbles.len(), nibbles_vec.len());
                prop_assert_eq!(nibbles.as_slice(), &nibbles_vec[..]);

                // Verify all nibbles are valid
                for (i, &nibble) in nibbles.iter().enumerate() {
                    prop_assert!(nibble <= 0xF, "Nibble at index {} is invalid: {}", i, nibble);
                }
            }

            /// Test `from_nibbles()` with invalid nibbles (values > 0xF)
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            #[should_panic(expected = "attempted to create invalid nibbles")]
            fn from_nibbles_invalid_panics(
                valid_nibbles in vec(0u8..=0xF, 0..50),
                invalid_nibble in 0x10u8..=0xFF,
                position in 0..=50usize
            ) {
                let mut nibbles_vec = valid_nibbles;
                let insert_pos = position.min(nibbles_vec.len());
                nibbles_vec.insert(insert_pos, invalid_nibble);

                // This should panic
                let _ = Nibbles::from_nibbles(&nibbles_vec);
            }

            /// Test `from_vec()` with valid nibbles
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn from_vec_valid(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles_vec_clone = nibbles_vec.clone();
                let nibbles = Nibbles::from_vec(nibbles_vec);
                prop_assert_eq!(nibbles.len(), nibbles_vec_clone.len());
                prop_assert_eq!(nibbles.as_slice(), &nibbles_vec_clone[..]);

                // Verify all nibbles are valid
                for (i, &nibble) in nibbles.iter().enumerate() {
                    prop_assert!(nibble <= 0xF, "Nibble at index {} is invalid: {}", i, nibble);
                }
            }

            /// Test `from_vec()` with invalid nibbles
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            #[should_panic(expected = "attempted to create invalid nibbles")]
            fn from_vec_invalid_panics(
                valid_nibbles in vec(0u8..=0xF, 0..50),
                invalid_nibble in 0x10u8..=0xFF,
                position in 0..=50usize
            ) {
                let mut nibbles_vec = valid_nibbles;
                let insert_pos = position.min(nibbles_vec.len());
                nibbles_vec.insert(insert_pos, invalid_nibble);

                // This should panic
                let _ = Nibbles::from_vec(nibbles_vec);
            }

            /// Test `unpack()` properly converts bytes to nibbles
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn unpack_byte_to_nibbles(bytes in vec(any::<u8>(), 0..100)) {
                let nibbles = Nibbles::unpack(&bytes);

                // Length should be exactly double
                prop_assert_eq!(nibbles.len(), bytes.len() * 2);

                // Each byte should be split into two nibbles
                for (i, &byte) in bytes.iter().enumerate() {
                    let high_nibble = byte >> 4;
                    let low_nibble = byte & 0x0F;

                    prop_assert_eq!(nibbles[i * 2], high_nibble,
                        "High nibble mismatch at byte {}: expected {}, got {}",
                        i, high_nibble, nibbles[i * 2]);
                    prop_assert_eq!(nibbles[i * 2 + 1], low_nibble,
                        "Low nibble mismatch at byte {}: expected {}, got {}",
                        i, low_nibble, nibbles[i * 2 + 1]);
                }

                // All resulting nibbles should be valid
                prop_assert!(valid_nibbles(&nibbles));
            }

            /// Test roundtrip: unpack -> pack should be identity
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn unpack_pack_identity(bytes in vec(any::<u8>(), 0..100)) {
                let nibbles = Nibbles::unpack(&bytes);
                let packed = nibbles.pack();
                prop_assert_eq!(&packed[..], &bytes[..],
                    "Roundtrip failed: unpack -> pack did not preserve original bytes");
            }

            /// Test roundtrip: from_nibbles -> pack -> unpack should preserve valid nibbles
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn nibbles_pack_unpack_roundtrip(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                // Only test with even-length nibbles for clean roundtrip
                let even_len = nibbles_vec.len() & !1;
                let nibbles_vec = &nibbles_vec[..even_len];

                if !nibbles_vec.is_empty() {
                    let nibbles = Nibbles::from_nibbles(nibbles_vec);
                    let packed = nibbles.pack();
                    let unpacked = Nibbles::unpack(&packed);

                    prop_assert_eq!(unpacked.as_slice(), nibbles_vec,
                        "Roundtrip failed: nibbles -> pack -> unpack did not preserve nibbles");
                }
            }

            /// Test that from_nibbles_unchecked accepts invalid nibbles without panic
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn from_nibbles_unchecked_accepts_invalid(nibbles_vec in vec(any::<u8>(), 0..100)) {
                // Should not panic even with invalid nibbles
                let nibbles = Nibbles::from_nibbles_unchecked(&nibbles_vec);
                prop_assert_eq!(nibbles.len(), nibbles_vec.len());
                prop_assert_eq!(nibbles.as_slice(), &nibbles_vec[..]);
            }

            /// Test that from_vec_unchecked accepts invalid nibbles without panic
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn from_vec_unchecked_accepts_invalid(nibbles_vec in vec(any::<u8>(), 0..100)) {
                let nibbles_vec_clone = nibbles_vec.clone();
                // Should not panic even with invalid nibbles
                let nibbles = Nibbles::from_vec_unchecked(nibbles_vec);
                prop_assert_eq!(nibbles.len(), nibbles_vec_clone.len());
                prop_assert_eq!(nibbles.as_slice(), &nibbles_vec_clone[..]);
            }

            /// Test pack() correctly combines nibbles into bytes
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn pack_combines_nibbles_correctly(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let packed = nibbles.pack();
                let expected_len = nibbles_vec.len().div_ceil(2);

                prop_assert_eq!(packed.len(), expected_len,
                    "Packed length should be (nibbles_len + 1) / 2");

                // Test each packed byte
                for i in 0..expected_len {
                    let byte_index = i * 2;
                    let expected_byte = if byte_index + 1 < nibbles_vec.len() {
                        // Even number of nibbles remaining: combine two nibbles
                        (nibbles_vec[byte_index] << 4) | nibbles_vec[byte_index + 1]
                    } else {
                        // Odd nibble at the end: shift left by 4
                        nibbles_vec[byte_index] << 4
                    };

                    prop_assert_eq!(packed[i], expected_byte,
                        "Packed byte at index {} should be 0x{:02x}, got 0x{:02x}",
                        i, expected_byte, packed[i]);
                }
            }

            /// Test to_vec() returns equivalent representation to as_slice()
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn to_vec_equivalent_to_slice(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let vec_result = nibbles.to_vec();
                let slice_result = nibbles.as_slice();

                prop_assert_eq!(vec_result.len(), slice_result.len());
                prop_assert_eq!(&vec_result[..], slice_result,
                    "to_vec() should return the same content as as_slice()");
                prop_assert_eq!(&vec_result[..], &nibbles_vec[..],
                    "to_vec() should return the original nibbles");
            }

            /// Test pack_to() works correctly for in-place packing
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn pack_to_in_place_packing(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let expected_packed = nibbles.pack();
                let expected_len = expected_packed.len();

                // Test with exact size buffer
                let mut buffer = vec![0u8; expected_len];
                nibbles.pack_to(&mut buffer);
                prop_assert_eq!(&buffer[..], &expected_packed[..],
                    "pack_to() with exact buffer should match pack()");

                // Test with larger buffer
                if expected_len > 0 {
                    let mut large_buffer = vec![0xFFu8; expected_len + 10];
                    nibbles.pack_to(&mut large_buffer[..expected_len]);
                    prop_assert_eq!(&large_buffer[..expected_len], &expected_packed[..],
                        "pack_to() with larger buffer should match pack()");
                    // Check that unused bytes weren't modified
                    prop_assert_eq!(&large_buffer[expected_len..], &vec![0xFFu8; 10][..],
                        "pack_to() should not modify bytes beyond the required range");
                }
            }

            /// Test roundtrip: nibbles -> pack -> unpack preserves data
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn nibbles_pack_unpack_roundtrip_all_lengths(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let original_nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let packed = original_nibbles.pack();
                let unpacked = Nibbles::unpack(&packed);

                if nibbles_vec.len() % 2 == 0 {
                    // Even length: perfect roundtrip
                    prop_assert_eq!(unpacked.as_slice(), &nibbles_vec[..],
                        "Even-length roundtrip should preserve all nibbles exactly");
                } else {
                    // Odd length: last nibble gets zero-padded
                    let expected_len = nibbles_vec.len() + 1;
                    prop_assert_eq!(unpacked.len(), expected_len,
                        "Odd-length roundtrip should add one padding nibble");
                    prop_assert_eq!(&unpacked[..nibbles_vec.len()], &nibbles_vec[..],
                        "Original nibbles should be preserved in odd-length roundtrip");
                    prop_assert_eq!(unpacked[nibbles_vec.len()], 0,
                        "Padding nibble should be zero in odd-length roundtrip");
                }
            }

            /// Test roundtrip: nibbles -> to_vec -> from_vec preserves data
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn nibbles_to_vec_from_vec_roundtrip(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let original_nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let vec_result = original_nibbles.to_vec();
                let roundtrip_nibbles = Nibbles::from_vec(vec_result);

                prop_assert_eq!(roundtrip_nibbles.len(), original_nibbles.len(),
                    "Roundtrip should preserve length");
                prop_assert_eq!(roundtrip_nibbles.as_slice(), original_nibbles.as_slice(),
                    "to_vec -> from_vec roundtrip should preserve all data");
                prop_assert_eq!(roundtrip_nibbles.as_slice(), &nibbles_vec[..],
                    "Roundtrip result should match original input");
            }

            /// Test invariant: packed length equals (nibbles length + 1) / 2
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn packed_length_invariant(nibbles_vec in vec(0u8..=0xF, 0..200)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let packed = nibbles.pack();
                let expected_packed_len = nibbles_vec.len().div_ceil(2);

                prop_assert_eq!(packed.len(), expected_packed_len,
                    "Packed length must be (nibbles_len + 1) / 2. \
                     nibbles_len={}, expected_packed_len={}, actual_packed_len={}",
                    nibbles_vec.len(), expected_packed_len, packed.len());

                // Test edge cases
                if nibbles_vec.is_empty() {
                    prop_assert_eq!(packed.len(), 0, "Empty nibbles should pack to empty");
                }
                if nibbles_vec.len() == 1 {
                    prop_assert_eq!(packed.len(), 1, "Single nibble should pack to one byte");
                }
                if nibbles_vec.len() == 2 {
                    prop_assert_eq!(packed.len(), 1, "Two nibbles should pack to one byte");
                }
                if nibbles_vec.len() == 3 {
                    prop_assert_eq!(packed.len(), 2, "Three nibbles should pack to two bytes");
                }
            }

            /// Test that pack_to correctly handles edge cases with odd/even lengths
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn pack_to_handles_odd_even_lengths(nibbles_vec in vec(0u8..=0xF, 0..50)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let expected_len = nibbles_vec.len().div_ceil(2);

                if expected_len > 0 {
                    let mut buffer = vec![0xAAu8; expected_len]; // Use non-zero pattern
                    nibbles.pack_to(&mut buffer);
                    let packed_direct = nibbles.pack();

                    prop_assert_eq!(&buffer[..], &packed_direct[..],
                        "pack_to should produce same result as pack for length {}",
                        nibbles_vec.len());

                    // Verify the last byte is correctly formed for odd lengths
                    if nibbles_vec.len() % 2 == 1 && !nibbles_vec.is_empty() {
                        let last_nibble = *nibbles_vec.last().unwrap();
                        let expected_last_byte = last_nibble << 4;
                        prop_assert_eq!(buffer[expected_len - 1], expected_last_byte,
                            "Last byte should be correctly formed for odd length: \
                             nibble=0x{:x}, expected_byte=0x{:02x}, actual_byte=0x{:02x}",
                            last_nibble, expected_last_byte, buffer[expected_len - 1]);
                    }
                }
            }

            /// Test conversion methods with empty input
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn conversion_methods_empty_input(_dummy in any::<u8>()) {
                let empty_nibbles = Nibbles::new();

                // Test pack() with empty input
                let packed = empty_nibbles.pack();
                prop_assert_eq!(packed.len(), 0, "Empty nibbles should pack to empty");

                // Test to_vec() with empty input
                let vec_result = empty_nibbles.to_vec();
                prop_assert_eq!(vec_result.len(), 0, "Empty nibbles to_vec should be empty");

                // Test pack_to() with empty input
                let mut buffer = vec![0xFFu8; 10];
                let original_buffer = buffer.clone();
                empty_nibbles.pack_to(&mut buffer[..0]); // Zero-length slice
                prop_assert_eq!(buffer, original_buffer,
                    "pack_to with empty nibbles should not modify buffer");
            }

            /// Test get() and indexing - verify bounds checking and correct values
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn get_and_indexing_bounds_and_values(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);

                // Test in-bounds access
                for (i, &expected) in nibbles_vec.iter().enumerate() {
                    prop_assert_eq!(nibbles.get(i), Some(&expected),
                        "get({}) should return Some(&{:x})", i, expected);
                    prop_assert_eq!(nibbles[i], expected,
                        "indexing [{}] should return {:x}", i, expected);
                }

                // Test out-of-bounds access
                prop_assert_eq!(nibbles.get(nibbles_vec.len()), None,
                    "get() should return None for index {}", nibbles_vec.len());
                prop_assert_eq!(nibbles.get(nibbles_vec.len() + 100), None,
                    "get() should return None for out-of-bounds index");
                prop_assert_eq!(nibbles.get(usize::MAX), None,
                    "get() should return None for usize::MAX");

                // Test empty nibbles
                let empty = Nibbles::new();
                prop_assert_eq!(empty.first(), None, "get(0) on empty nibbles should return None");
            }

            /// Test get_byte() and get_byte_unchecked() - verify byte reconstruction from nibble pairs
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn get_byte_methods_reconstruct_correctly(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);

                // Test in-bounds get_byte()
                for i in 0..nibbles_vec.len().saturating_sub(1) {
                    let expected_byte = (nibbles_vec[i] << 4) | nibbles_vec[i + 1];
                    prop_assert_eq!(nibbles.get_byte(i), Some(expected_byte),
                        "get_byte({}) should combine nibbles {:x} and {:x} to {:02x}",
                        i, nibbles_vec[i], nibbles_vec[i + 1], expected_byte);

                    // Test get_byte_unchecked() for same index
                    unsafe {
                        prop_assert_eq!(nibbles.get_byte_unchecked(i), expected_byte,
                            "get_byte_unchecked({}) should return {:02x}", i, expected_byte);
                    }
                }

                // Test out-of-bounds get_byte()
                if !nibbles_vec.is_empty() {
                    prop_assert_eq!(nibbles.get_byte(nibbles_vec.len() - 1), None,
                        "get_byte() at last valid index should return None (needs pair)");
                }
                prop_assert_eq!(nibbles.get_byte(nibbles_vec.len()), None,
                    "get_byte() should return None for out-of-bounds index");
                prop_assert_eq!(nibbles.get_byte(usize::MAX), None,
                    "get_byte() should return None for usize::MAX");

                // Test edge case: single nibble
                if nibbles_vec.len() == 1 {
                    prop_assert_eq!(nibbles.get_byte(0), None,
                        "get_byte(0) with single nibble should return None");
                }

                // Test empty nibbles
                let empty = Nibbles::new();
                prop_assert_eq!(empty.get_byte(0), None, "get_byte(0) on empty nibbles should return None");
            }

            /// Test first() and last() - verify they return correct boundary elements
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn first_and_last_boundary_elements(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);

                if nibbles_vec.is_empty() {
                    prop_assert_eq!(nibbles.first(), None, "first() on empty nibbles should return None");
                    prop_assert_eq!(nibbles.last(), None, "last() on empty nibbles should return None");
                } else {
                    prop_assert_eq!(nibbles.first(), Some(nibbles_vec[0]),
                        "first() should return the first element: {:x}", nibbles_vec[0]);
                    prop_assert_eq!(nibbles.last(), Some(*nibbles_vec.last().unwrap()),
                        "last() should return the last element: {:x}", nibbles_vec.last().unwrap());
                }

                // Test single element
                if nibbles_vec.len() == 1 {
                    prop_assert_eq!(nibbles.first(), nibbles.last(),
                        "first() and last() should be equal for single-element nibbles");
                }
            }

            /// Test set_at() and set_at_unchecked() - verify modification preserves other elements
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn set_at_methods_preserve_other_elements(
                nibbles_vec in vec(0u8..=0xF, 1..100),
                index in 0..100usize,
                new_value in 0u8..=0xF
            ) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original = nibbles.clone();

                if index < nibbles_vec.len() {
                    // Test set_at() - in bounds
                    nibbles.set_at(index, new_value);

                    // Verify the target index was changed
                    prop_assert_eq!(nibbles[index], new_value,
                        "set_at({}, {:x}) should update the value at index", index, new_value);

                    // Verify all other elements are unchanged
                    for i in 0..nibbles_vec.len() {
                        if i != index {
                            prop_assert_eq!(nibbles[i], original[i],
                                "set_at() should not modify element at index {} (was {:x}, now {:x})",
                                i, original[i], nibbles[i]);
                        }
                    }

                    // Test set_at_unchecked() - reset and test again
                    let mut nibbles_unchecked = original.clone();
                    nibbles_unchecked.set_at_unchecked(index, new_value);
                    prop_assert_eq!(nibbles_unchecked[index], new_value,
                        "set_at_unchecked({}, {:x}) should update the value", index, new_value);

                    // Verify other elements unchanged for unchecked version
                    for i in 0..nibbles_vec.len() {
                        if i != index {
                            prop_assert_eq!(nibbles_unchecked[i], original[i],
                                "set_at_unchecked() should not modify other elements");
                        }
                    }
                }
            }

            /// Test set_at() bounds checking and invalid value checking
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            #[should_panic]
            fn set_at_panics_on_invalid_value(
                nibbles_vec in vec(0u8..=0xF, 1..10),
                invalid_value in 0x10u8..=0xFF
            ) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                nibbles.set_at(0, invalid_value); // Should panic
            }

            /// Test set_at() bounds checking for out-of-bounds index
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            #[should_panic]
            fn set_at_panics_on_out_of_bounds(
                nibbles_vec in vec(0u8..=0xF, 1..10),
                valid_value in 0u8..=0xF
            ) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                nibbles.set_at(nibbles_vec.len(), valid_value); // Should panic - out of bounds
            }

            /// Test at() method - verify it returns correct nibble values
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn at_method_returns_correct_values(nibbles_vec in vec(0u8..=0xF, 1..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);

                // Test all valid indices
                for (i, &expected) in nibbles_vec.iter().enumerate() {
                    prop_assert_eq!(nibbles.at(i), expected as usize,
                        "at({}) should return {} (as usize)", i, expected);
                }
            }

            /// Test at() method bounds checking
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            #[should_panic]
            fn at_method_panics_on_out_of_bounds(nibbles_vec in vec(0u8..=0xF, 1..10)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let _ = nibbles.at(nibbles_vec.len()); // Should panic
            }

            /// Test edge cases for empty nibbles with access methods
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn access_methods_empty_nibbles_edge_cases(_dummy in any::<u8>()) {
                let empty = Nibbles::new();

                // All get methods should return None/appropriate empty response
                prop_assert_eq!(empty.first(), None, "get(0) on empty should return None");
                prop_assert_eq!(empty.get_byte(0), None, "get_byte(0) on empty should return None");
                prop_assert_eq!(empty.first(), None, "first() on empty should return None");
                prop_assert_eq!(empty.last(), None, "last() on empty should return None");
            }

            /// Test boundary indices for get_byte() method
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn get_byte_boundary_indices(nibbles_vec in vec(0u8..=0xF, 2..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);

                // Test first valid index
                if nibbles_vec.len() >= 2 {
                    let expected = (nibbles_vec[0] << 4) | nibbles_vec[1];
                    prop_assert_eq!(nibbles.get_byte(0), Some(expected),
                        "get_byte(0) should work for nibbles of length >= 2");
                }

                // Test last valid index
                if nibbles_vec.len() >= 2 {
                    let last_valid_idx = nibbles_vec.len() - 2;
                    let expected = (nibbles_vec[last_valid_idx] << 4) | nibbles_vec[last_valid_idx + 1];
                    prop_assert_eq!(nibbles.get_byte(last_valid_idx), Some(expected),
                        "get_byte() should work at last valid index");
                }

                // Test first invalid index (needs two nibbles)
                if !nibbles_vec.is_empty() {
                    let first_invalid_idx = nibbles_vec.len() - 1;
                    prop_assert_eq!(nibbles.get_byte(first_invalid_idx), None,
                        "get_byte() should return None when only one nibble remains");
                }
            }

            // ================================================================================
            // Property tests for Nibbles manipulation methods
            // ================================================================================

            /// Test push() and push_unchecked() - verify length increases and element is added correctly
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn push_methods_length_and_correctness(
                nibbles_vec in vec(0u8..=0xF, 0..50),
                new_nibble in 0u8..=0xF
            ) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original_len = nibbles.len();

                // Test push()
                nibbles.push(new_nibble);

                // Verify length increased by 1
                prop_assert_eq!(nibbles.len(), original_len + 1,
                    "push() should increase length by 1");

                // Verify new element was added at the end
                prop_assert_eq!(nibbles[original_len], new_nibble,
                    "push() should add new nibble at the end");

                // Verify existing elements are preserved
                for i in 0..original_len {
                    prop_assert_eq!(nibbles[i], nibbles_vec[i],
                        "push() should preserve existing elements at index {}", i);
                }

                // Test push_unchecked() with a fresh copy
                let mut nibbles_unchecked = Nibbles::from_nibbles(&nibbles_vec);
                nibbles_unchecked.push_unchecked(new_nibble);

                // Should behave identically to push() for valid nibbles
                prop_assert_eq!(nibbles_unchecked.len(), original_len + 1,
                    "push_unchecked() should increase length by 1");
                prop_assert_eq!(nibbles_unchecked[original_len], new_nibble,
                    "push_unchecked() should add new nibble at the end");

                // Test that both methods produce identical results for valid input
                prop_assert_eq!(nibbles.as_slice(), nibbles_unchecked.as_slice(),
                    "push() and push_unchecked() should produce identical results for valid nibbles");
            }

            /// Test push() with invalid nibbles - should panic
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            #[should_panic]
            fn push_invalid_nibble_panics(
                nibbles_vec in vec(0u8..=0xF, 0..10),
                invalid_nibble in 0x10u8..=0xFF
            ) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                nibbles.push(invalid_nibble); // Should panic
            }

            /// Test push_unchecked() accepts invalid nibbles without panic
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn push_unchecked_accepts_invalid(
                nibbles_vec in vec(0u8..=0xF, 0..50),
                invalid_nibble in 0x10u8..=0xFF
            ) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original_len = nibbles.len();

                // Should not panic even with invalid nibbles
                nibbles.push_unchecked(invalid_nibble);

                prop_assert_eq!(nibbles.len(), original_len + 1,
                    "push_unchecked() should increase length even with invalid nibbles");
                prop_assert_eq!(nibbles[original_len], invalid_nibble,
                    "push_unchecked() should add invalid nibble at the end");
            }

            /// Test pop() - verify length decreases and correct element is returned
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn pop_method_length_and_correctness(nibbles_vec in vec(0u8..=0xF, 0..50)) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original_len = nibbles.len();

                if nibbles_vec.is_empty() {
                    // Test pop() on empty nibbles
                    prop_assert_eq!(nibbles.pop(), None,
                        "pop() on empty nibbles should return None");
                    prop_assert_eq!(nibbles.len(), 0,
                        "pop() on empty nibbles should not change length");
                } else {
                    let expected_popped = *nibbles_vec.last().unwrap();

                    // Test pop()
                    let popped = nibbles.pop();

                    // Verify correct element was returned
                    prop_assert_eq!(popped, Some(expected_popped),
                        "pop() should return the last element");

                    // Verify length decreased by 1
                    prop_assert_eq!(nibbles.len(), original_len - 1,
                        "pop() should decrease length by 1");

                    // Verify remaining elements are preserved
                    for i in 0..nibbles.len() {
                        prop_assert_eq!(nibbles[i], nibbles_vec[i],
                            "pop() should preserve remaining elements at index {}", i);
                    }
                }
            }

            /// Test push/pop roundtrip - verify they are inverse operations
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn push_pop_roundtrip(
                nibbles_vec in vec(0u8..=0xF, 0..50),
                new_nibble in 0u8..=0xF
            ) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original = nibbles.clone();

                // Push then pop should restore original state
                nibbles.push(new_nibble);
                let popped = nibbles.pop();

                prop_assert_eq!(popped, Some(new_nibble),
                    "pop() should return the nibble that was just pushed");
                prop_assert_eq!(nibbles.as_slice(), original.as_slice(),
                    "push() then pop() should restore original state");
            }

            /// Test extend_from_slice() - verify concatenation with valid Nibbles
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn extend_from_slice_method_concatenation(
                first_vec in vec(0u8..=0xF, 0..50),
                second_vec in vec(0u8..=0xF, 0..50)
            ) {
                let mut nibbles = Nibbles::from_nibbles(&first_vec);
                let other_nibbles = Nibbles::from_nibbles(&second_vec);
                let original_len = nibbles.len();

                // Extend with other nibbles
                nibbles.extend_from_slice(&other_nibbles);

                // Verify total length
                prop_assert_eq!(nibbles.len(), original_len + second_vec.len(),
                    "extend_from_slice() should increase length by the length of added nibbles");

                // Verify original elements preserved
                for i in 0..original_len {
                    prop_assert_eq!(nibbles[i], first_vec[i],
                        "extend_from_slice() should preserve original elements at index {}", i);
                }

                // Verify new elements added correctly
                for i in 0..second_vec.len() {
                    prop_assert_eq!(nibbles[original_len + i], second_vec[i],
                        "extend_from_slice() should add new elements correctly at index {}", original_len + i);
                }

                // Verify result equals concatenation
                let mut expected = first_vec;
                expected.extend_from_slice(&second_vec);
                prop_assert_eq!(nibbles.as_slice(), &expected[..],
                    "extend_from_slice() result should equal direct concatenation");
            }

            /// Test extend_from_slice_unchecked() - verify concatenation with any byte slice
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn extend_from_slice_unchecked_method(
                first_vec in vec(0u8..=0xF, 0..50),
                second_vec in vec(any::<u8>(), 0..50)
            ) {
                let mut nibbles = Nibbles::from_nibbles(&first_vec);
                let original_len = nibbles.len();

                // Extend with arbitrary bytes (including invalid nibbles)
                nibbles.extend_from_slice_unchecked(&second_vec);

                // Verify total length
                prop_assert_eq!(nibbles.len(), original_len + second_vec.len(),
                    "extend_from_slice_unchecked() should increase length correctly");

                // Verify original elements preserved
                for i in 0..original_len {
                    prop_assert_eq!(nibbles[i], first_vec[i],
                        "extend_from_slice_unchecked() should preserve original elements at index {}", i);
                }

                // Verify new elements added correctly (even if invalid)
                for i in 0..second_vec.len() {
                    prop_assert_eq!(nibbles[original_len + i], second_vec[i],
                        "extend_from_slice_unchecked() should add new elements at index {}", original_len + i);
                }
            }

            /// Test extend methods with empty inputs
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn extend_methods_empty_inputs(nibbles_vec in vec(0u8..=0xF, 0..50)) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original = nibbles.clone();

                // Extend with empty Nibbles
                let empty_nibbles = Nibbles::new();
                nibbles.extend_from_slice(&empty_nibbles);

                prop_assert_eq!(nibbles.as_slice(), original.as_slice(),
                    "extend_from_slice() with empty nibbles should not change original");

                // Extend empty nibbles with non-empty
                let mut empty = Nibbles::new();
                empty.extend_from_slice(&original);

                prop_assert_eq!(empty.as_slice(), original.as_slice(),
                    "extend_from_slice() from empty should equal the added nibbles");

                // Test extend_from_slice_unchecked with empty slice
                let mut nibbles_unchecked = original.clone();
                nibbles_unchecked.extend_from_slice_unchecked(&[]);

                prop_assert_eq!(nibbles_unchecked.as_slice(), original.as_slice(),
                    "extend_from_slice_unchecked() with empty slice should not change original");
            }

            /// Test truncate() - verify prefix preservation and length adjustment
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn truncate_method_prefix_preservation(
                nibbles_vec in vec(0u8..=0xF, 0..100),
                new_len in 0..100usize
            ) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original_len = nibbles.len();
                let effective_new_len = new_len.min(original_len);

                // Store expected prefix
                let expected_prefix = &nibbles_vec[..effective_new_len];

                // Truncate
                nibbles.truncate(new_len);

                // Verify new length
                prop_assert_eq!(nibbles.len(), effective_new_len,
                    "truncate({}) should set length to {}", new_len, effective_new_len);

                // Verify prefix is preserved
                prop_assert_eq!(nibbles.as_slice(), expected_prefix,
                    "truncate() should preserve prefix of original nibbles");

                // Verify each element individually
                for i in 0..effective_new_len {
                    prop_assert_eq!(nibbles[i], nibbles_vec[i],
                        "truncate() should preserve element at index {}", i);
                }
            }

            /// Test truncate() with various edge cases
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn truncate_edge_cases(nibbles_vec in vec(0u8..=0xF, 1..50)) {
                // Test truncate to 0 (clear)
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                nibbles.truncate(0);

                prop_assert_eq!(nibbles.len(), 0,
                    "truncate(0) should result in empty nibbles");
                prop_assert!(nibbles.is_empty(),
                    "truncate(0) should result in empty nibbles");

                // Test truncate to same length (no-op)
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original_len = nibbles.len();
                nibbles.truncate(original_len);

                prop_assert_eq!(nibbles.len(), original_len,
                    "truncate(len) should not change length");
                prop_assert_eq!(nibbles.as_slice(), &nibbles_vec[..],
                    "truncate(len) should not change contents");

                // Test truncate to larger than current length (no-op)
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                nibbles.truncate(original_len + 100);

                prop_assert_eq!(nibbles.len(), original_len,
                    "truncate(len + 100) should not change length");
                prop_assert_eq!(nibbles.as_slice(), &nibbles_vec[..],
                    "truncate(len + 100) should not change contents");
            }

            /// Test slice() - verify correct subsequence extraction
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn slice_method_subsequence_extraction(
                nibbles_vec in vec(0u8..=0xF, 0..50),
                start in 0..50usize,
                end in 0..50usize
            ) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let len = nibbles.len();

                // Ensure valid range
                let start = start.min(len);
                let end = end.max(start).min(len);

                if start <= end {
                    // Test slicing with Range
                    let sliced = nibbles.slice(start..end);
                    let expected_len = end - start;

                    prop_assert_eq!(sliced.len(), expected_len,
                        "slice({}..{}) should have length {}", start, end, expected_len);

                    // Verify contents match expected subsequence
                    for i in 0..expected_len {
                        prop_assert_eq!(sliced[i], nibbles_vec[start + i],
                            "slice()[{}] should equal original[{}]", i, start + i);
                    }

                    // Test that sliced nibbles are valid
                    prop_assert!(valid_nibbles(&sliced),
                        "slice() result should contain valid nibbles");

                    // Test equivalence with direct indexing
                    if expected_len > 0 {
                        prop_assert_eq!(sliced.as_slice(), &nibbles_vec[start..end],
                            "slice() should match direct slice of original vector");
                    }
                }
            }

            /// Test slice() with various range types
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn slice_various_range_types(nibbles_vec in vec(0u8..=0xF, 5..20)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let len = nibbles.len();

                if len >= 5 {
                    // Test RangeInclusive
                    let sliced_inclusive = nibbles.slice(1..=3);
                    prop_assert_eq!(sliced_inclusive.len(), 3,
                        "slice(1..=3) should have length 3");
                    prop_assert_eq!(sliced_inclusive.as_slice(), &nibbles_vec[1..=3],
                        "slice(1..=3) should match inclusive range");

                    // Test RangeFrom
                    let sliced_from = nibbles.slice(2..);
                    prop_assert_eq!(sliced_from.len(), len - 2,
                        "slice(2..) should have length {}", len - 2);
                    prop_assert_eq!(sliced_from.as_slice(), &nibbles_vec[2..],
                        "slice(2..) should match range from");

                    // Test RangeTo
                    let sliced_to = nibbles.slice(..3);
                    prop_assert_eq!(sliced_to.len(), 3,
                        "slice(..3) should have length 3");
                    prop_assert_eq!(sliced_to.as_slice(), &nibbles_vec[..3],
                        "slice(..3) should match range to");

                    // Test RangeFull
                    let sliced_full = nibbles.slice(..);
                    prop_assert_eq!(sliced_full.len(), len,
                        "slice(..) should have same length as original");
                    prop_assert_eq!(sliced_full.as_slice(), nibbles.as_slice(),
                        "slice(..) should match full original");
                }
            }

            /// Test slice() edge cases
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn slice_edge_cases(nibbles_vec in vec(0u8..=0xF, 0..20)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let len = nibbles.len();

                // Test empty slice
                let empty_slice = nibbles.slice(0..0);
                prop_assert_eq!(empty_slice.len(), 0,
                    "slice(0..0) should be empty");
                prop_assert!(empty_slice.is_empty(),
                    "slice(0..0) should be empty");

                // Test single element slice (if possible)
                if len >= 1 {
                    let single_slice = nibbles.slice(0..1);
                    prop_assert_eq!(single_slice.len(), 1,
                        "slice(0..1) should have length 1");
                    prop_assert_eq!(single_slice[0], nibbles_vec[0],
                        "slice(0..1)[0] should equal original[0]");
                }

                // Test slice at end
                if len >= 1 {
                    let end_slice = nibbles.slice(len-1..len);
                    prop_assert_eq!(end_slice.len(), 1,
                        "slice at end should have length 1");
                    prop_assert_eq!(end_slice[0], nibbles_vec[len-1],
                        "slice at end should contain last element");
                }
            }

            /// Test join() - verify concatenation of two Nibbles
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn join_method_concatenation(
                first_vec in vec(0u8..=0xF, 0..50),
                second_vec in vec(0u8..=0xF, 0..50)
            ) {
                let first_nibbles = Nibbles::from_nibbles(&first_vec);
                let second_nibbles = Nibbles::from_nibbles(&second_vec);

                let joined = first_nibbles.join(&second_nibbles);

                // Verify total length
                prop_assert_eq!(joined.len(), first_vec.len() + second_vec.len(),
                    "join() should have combined length");

                // Verify first part matches first nibbles
                for i in 0..first_vec.len() {
                    prop_assert_eq!(joined[i], first_vec[i],
                        "join() should preserve first nibbles at index {}", i);
                }

                // Verify second part matches second nibbles
                for i in 0..second_vec.len() {
                    prop_assert_eq!(joined[first_vec.len() + i], second_vec[i],
                        "join() should append second nibbles at index {}", first_vec.len() + i);
                }

                // Verify result equals direct concatenation
                let mut expected = first_vec.clone();
                expected.extend_from_slice(&second_vec);
                prop_assert_eq!(joined.as_slice(), &expected[..],
                    "join() result should equal direct concatenation");

                // Verify original nibbles are unchanged
                prop_assert_eq!(first_nibbles.as_slice(), &first_vec[..],
                    "join() should not modify first nibbles");
                prop_assert_eq!(second_nibbles.as_slice(), &second_vec[..],
                    "join() should not modify second nibbles");
            }

            /// Test join() edge cases and properties
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn join_edge_cases_and_properties(nibbles_vec in vec(0u8..=0xF, 0..50)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let empty = Nibbles::new();

                // Test join with empty (right identity)
                let joined_with_empty = nibbles.join(&empty);
                prop_assert_eq!(joined_with_empty.as_slice(), nibbles.as_slice(),
                    "join() with empty should be identity (right)");

                // Test empty join with non-empty (left identity)
                let empty_joined = empty.join(&nibbles);
                prop_assert_eq!(empty_joined.as_slice(), nibbles.as_slice(),
                    "empty join() should be identity (left)");

                // Test join two empty nibbles
                let empty_join_empty = empty.join(&Nibbles::new());
                prop_assert_eq!(empty_join_empty.len(), 0,
                    "empty join empty should be empty");
                prop_assert!(empty_join_empty.is_empty(),
                    "empty join empty should be empty");

                // Test self-join
                let self_joined = nibbles.join(&nibbles);
                prop_assert_eq!(self_joined.len(), nibbles.len() * 2,
                    "self-join should double the length");

                if !nibbles_vec.is_empty() {
                    // Verify first half equals original
                    for i in 0..nibbles_vec.len() {
                        prop_assert_eq!(self_joined[i], nibbles_vec[i],
                            "self-join first half should match original at index {}", i);
                    }
                    // Verify second half equals original
                    for i in 0..nibbles_vec.len() {
                        prop_assert_eq!(self_joined[nibbles_vec.len() + i], nibbles_vec[i],
                            "self-join second half should match original at index {}", i);
                    }
                }
            }

            /// Test join() associativity property
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn join_associativity(
                a_vec in vec(0u8..=0xF, 0..20),
                b_vec in vec(0u8..=0xF, 0..20),
                c_vec in vec(0u8..=0xF, 0..20)
            ) {
                let a = Nibbles::from_nibbles(&a_vec);
                let b = Nibbles::from_nibbles(&b_vec);
                let c = Nibbles::from_nibbles(&c_vec);

                // Test (a.join(b)).join(c) == a.join(b.join(c))
                let left_assoc = a.join(&b).join(&c);
                let right_assoc = a.join(&b.join(&c));

                prop_assert_eq!(left_assoc.as_slice(), right_assoc.as_slice(),
                    "join() should be associative: (a.join(b)).join(c) == a.join(b.join(c))");
            }

            /// Test clear() - verify all elements are removed
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn clear_method_removes_all_elements(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);
                let original_len = nibbles.len();

                // Clear the nibbles
                nibbles.clear();

                // Verify length is 0
                prop_assert_eq!(nibbles.len(), 0,
                    "clear() should set length to 0");

                // Verify is_empty returns true
                prop_assert!(nibbles.is_empty(),
                    "clear() should make nibbles empty");

                // Verify as_slice is empty
                prop_assert_eq!(nibbles.as_slice(), &[],
                    "clear() should result in empty slice");

                // Verify first() and last() return None
                prop_assert_eq!(nibbles.first(), None,
                    "clear() should make first() return None");
                prop_assert_eq!(nibbles.last(), None,
                    "clear() should make last() return None");

                // Verify capacity may be preserved (implementation detail)
                // but we can still push after clear
                if original_len > 0 {
                    nibbles.push(0xA);
                    prop_assert_eq!(nibbles.len(), 1,
                        "should be able to push after clear()");
                    prop_assert_eq!(nibbles[0], 0xA,
                        "pushed element after clear() should be correct");
                }
            }

            /// Test clear() on already empty nibbles
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn clear_on_empty_nibbles(_dummy in any::<u8>()) {
                let mut empty = Nibbles::new();

                // Clear already empty nibbles
                empty.clear();

                // Should remain empty
                prop_assert_eq!(empty.len(), 0,
                    "clear() on empty nibbles should remain empty");
                prop_assert!(empty.is_empty(),
                    "clear() on empty nibbles should remain empty");
                prop_assert_eq!(empty.as_slice(), &[],
                    "clear() on empty nibbles should have empty slice");
            }

            /// Test multiple clears in sequence
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn multiple_clears_sequence(nibbles_vec in vec(0u8..=0xF, 1..50)) {
                let mut nibbles = Nibbles::from_nibbles(&nibbles_vec);

                // Clear multiple times
                nibbles.clear();
                nibbles.clear();
                nibbles.clear();

                // Should still be empty
                prop_assert_eq!(nibbles.len(), 0,
                    "multiple clear() calls should result in empty nibbles");
                prop_assert!(nibbles.is_empty(),
                    "multiple clear() calls should result in empty nibbles");

                // Should still be able to use after multiple clears
                nibbles.push(0xF);
                prop_assert_eq!(nibbles.len(), 1,
                    "should be able to push after multiple clear() calls");
                prop_assert_eq!(nibbles[0], 0xF,
                    "element added after multiple clears should be correct");
            }

            /// Test roundtrip properties combining multiple methods
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn manipulation_methods_roundtrip_properties(
                base_vec in vec(0u8..=0xF, 5..30),
                extension in vec(0u8..=0xF, 0..20),
                _truncate_len in 1..15usize
            ) {
                let mut nibbles = Nibbles::from_nibbles(&base_vec);
                let original = nibbles.clone();

                // Extend then truncate back to original length
                nibbles.extend_from_slice(&Nibbles::from_nibbles(&extension));
                nibbles.truncate(base_vec.len());

                prop_assert_eq!(nibbles.as_slice(), original.as_slice(),
                    "extend then truncate to original length should restore original");

                // Clear then extend should equal just the extension
                let mut cleared_nibbles = original.clone();
                cleared_nibbles.clear();
                cleared_nibbles.extend_from_slice(&Nibbles::from_nibbles(&extension));

                let extension_nibbles = Nibbles::from_nibbles(&extension);
                prop_assert_eq!(cleared_nibbles.as_slice(), extension_nibbles.as_slice(),
                    "clear then extend should equal the extension");

                // Slice then join should preserve data
                if base_vec.len() > 5 {
                    let mid = base_vec.len() / 2;
                    let first_half = original.slice(..mid);
                    let second_half = original.slice(mid..);
                    let rejoined = first_half.join(&second_half);

                    prop_assert_eq!(rejoined.as_slice(), original.as_slice(),
                        "slice then join should preserve original data");
                }
            }

            /// Test starts_with() and has_prefix() - verify prefix checking with transitivity and reflexivity
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn starts_with_and_has_prefix_properties(
                a_vec in vec(0u8..=0xF, 0..50),
                b_vec in vec(0u8..=0xF, 0..30),
                c_vec in vec(0u8..=0xF, 0..20)
            ) {
                let a = Nibbles::from_nibbles(&a_vec);
                let b = Nibbles::from_nibbles(&b_vec);
                let _c = Nibbles::from_nibbles(&c_vec);

                // Test reflexivity: every nibbles starts with itself
                prop_assert!(a.starts_with(&a_vec),
                    "reflexivity: nibbles should start with itself");
                prop_assert!(a.has_prefix(&a_vec),
                    "reflexivity: has_prefix should be true for self");

                // Test that has_prefix is equivalent to starts_with
                prop_assert_eq!(a.starts_with(&b_vec), a.has_prefix(&b_vec),
                    "has_prefix should be equivalent to starts_with");

                // Test transitivity: if a starts with b and b starts with c, then a starts with c
                if a.starts_with(&b_vec) && b.starts_with(&c_vec) {
                    prop_assert!(a.starts_with(&c_vec),
                        "transitivity: if a starts with b and b starts with c, then a starts with c");
                }

                // Test empty prefix: everything starts with empty
                let empty = Nibbles::new();
                prop_assert!(a.starts_with(empty.as_slice()),
                    "every nibbles should start with empty nibbles");
                prop_assert!(a.has_prefix(empty.as_slice()),
                    "every nibbles should have empty prefix");

                // Test prefix length relationship
                if a.starts_with(&b_vec) {
                    prop_assert!(b_vec.len() <= a_vec.len(),
                        "if a starts with b, then b.len() <= a.len()");
                    prop_assert_eq!(&a_vec[..b_vec.len()], &b_vec[..],
                        "if a starts with b, then a[..b.len()] == b");
                }

                // Test that longer prefixes cannot be prefixes of shorter sequences
                if b_vec.len() > a_vec.len() && !a_vec.is_empty() {
                    prop_assert!(!a.starts_with(&b_vec),
                        "longer sequence cannot be prefix of shorter sequence");
                }
            }

            /// Test ends_with() - verify suffix checking
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn ends_with_properties(
                a_vec in vec(0u8..=0xF, 0..50),
                b_vec in vec(0u8..=0xF, 0..30)
            ) {
                let a = Nibbles::from_nibbles(&a_vec);
                let b = Nibbles::from_nibbles(&b_vec);

                // Test reflexivity: every nibbles ends with itself
                prop_assert!(a.ends_with(&a_vec),
                    "reflexivity: nibbles should end with itself");

                // Test empty suffix: everything ends with empty
                let empty = Nibbles::new();
                prop_assert!(a.ends_with(empty.as_slice()),
                    "every nibbles should end with empty nibbles");

                // Test suffix length relationship
                if a.ends_with(&b_vec) && !b_vec.is_empty() {
                    prop_assert!(b_vec.len() <= a_vec.len(),
                        "if a ends with b, then b.len() <= a.len()");
                    prop_assert_eq!(&a_vec[a_vec.len() - b_vec.len()..], &b_vec[..],
                        "if a ends with b, then a[a.len()-b.len()..] == b");
                }

                // Test that longer suffixes cannot be suffixes of shorter sequences
                if b_vec.len() > a_vec.len() && !a_vec.is_empty() {
                    prop_assert!(!a.ends_with(&b_vec),
                        "longer sequence cannot be suffix of shorter sequence");
                }

                // Test transitivity for ends_with
                // For this test, we'll check transitivity by creating a suffix of b_vec if possible
                if !b_vec.is_empty() && b_vec.len() >= 2 {
                    let c_vec = &b_vec[b_vec.len() / 2..];
                    if a.ends_with(&b_vec) && b.ends_with(c_vec) {
                        prop_assert!(a.ends_with(c_vec),
                            "transitivity: if a ends with b and b ends with c, then a ends with c");
                    }
                }
            }

            /// Test common_prefix_length() - verify symmetry and correctness
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn common_prefix_length_properties(
                a_vec in vec(0u8..=0xF, 0..50),
                b_vec in vec(0u8..=0xF, 0..50)
            ) {
                let a = Nibbles::from_nibbles(&a_vec);
                let b = Nibbles::from_nibbles(&b_vec);

                // Test symmetry: common_prefix_length(a, b) == common_prefix_length(b, a)
                prop_assert_eq!(a.common_prefix_length(&b_vec), b.common_prefix_length(&a_vec),
                    "common_prefix_length should be symmetric");

                // Test reflexivity: common_prefix_length(a, a) == a.len()
                prop_assert_eq!(a.common_prefix_length(&a_vec), a_vec.len(),
                    "common_prefix_length with self should equal length");

                // Test with empty: common_prefix_length(a, empty) == 0
                let empty = Nibbles::new();
                prop_assert_eq!(a.common_prefix_length(empty.as_slice()), 0,
                    "common_prefix_length with empty should be 0");
                prop_assert_eq!(empty.common_prefix_length(&a_vec), 0,
                    "empty's common_prefix_length with anything should be 0");

                // Test bounds: result <= min(a.len(), b.len())
                let result = a.common_prefix_length(&b_vec);
                let min_len = a_vec.len().min(b_vec.len());
                prop_assert!(result <= min_len,
                    "common_prefix_length should be <= min(a.len(), b.len()): {} <= {}",
                    result, min_len);

                // Test correctness: first 'result' elements should match
                if result > 0 {
                    prop_assert_eq!(&a_vec[..result], &b_vec[..result],
                        "first {} elements should match", result);
                }

                // Test correctness: element at 'result' should differ (if both sequences are long enough)
                if result < a_vec.len() && result < b_vec.len() {
                    prop_assert_ne!(a_vec[result], b_vec[result],
                        "elements at position {} should differ: {:x} != {:x}",
                        result, a_vec[result], b_vec[result]);
                }

                // Test monotonicity with prefix extension
                if !a_vec.is_empty() && !b_vec.is_empty() {
                    let a_prefix = Nibbles::from_nibbles(&a_vec[..a_vec.len().min(1)]);
                    let b_prefix = Nibbles::from_nibbles(&b_vec[..b_vec.len().min(1)]);
                    let prefix_common = a_prefix.common_prefix_length(b_prefix.as_slice());

                    if prefix_common > 0 {
                        prop_assert!(result >= prefix_common,
                            "common prefix length should be monotonic with extensions");
                    }
                }
            }

            /// Test len() and is_empty() - verify length and emptiness queries
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn len_and_is_empty_properties(nibbles_vec in vec(0u8..=0xF, 0..100)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);

                // Test len() correctness
                prop_assert_eq!(nibbles.len(), nibbles_vec.len(),
                    "len() should return the number of nibbles");

                // Test is_empty() correctness
                prop_assert_eq!(nibbles.is_empty(), nibbles_vec.is_empty(),
                    "is_empty() should match the underlying vector's emptiness");

                // Test len() and is_empty() relationship
                prop_assert_eq!(nibbles.is_empty(), nibbles.is_empty(),
                    "is_empty() should be equivalent to len() == 0");

                // Test as_slice().len() consistency
                prop_assert_eq!(nibbles.as_slice().len(), nibbles.len(),
                    "as_slice().len() should match len()");

                // Test specific empty case
                if nibbles_vec.is_empty() {
                    prop_assert!(nibbles.is_empty(),
                        "nibbles created from empty vec should be empty");
                    prop_assert_eq!(nibbles.len(), 0,
                        "empty nibbles should have length 0");
                } else {
                    prop_assert!(!nibbles.is_empty(),
                        "nibbles created from non-empty vec should not be empty");
                    prop_assert!(!nibbles.is_empty(),
                        "non-empty nibbles should have positive length");
                }

                // Test new() creates empty nibbles
                let new_nibbles = Nibbles::new();
                prop_assert!(new_nibbles.is_empty(),
                    "Nibbles::new() should create empty nibbles");
                prop_assert_eq!(new_nibbles.len(), 0,
                    "Nibbles::new() should have length 0");
            }

            /// Test length consistency across all operations
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn length_consistency_across_operations(
                base_vec in vec(0u8..=0xF, 0..50),
                other_vec in vec(0u8..=0xF, 0..30),
                slice_start in 0..25usize,
                slice_end in 0..25usize
            ) {
                let nibbles = Nibbles::from_nibbles(&base_vec);
                let other = Nibbles::from_nibbles(&other_vec);

                // Test pack() length consistency: packed.len() == (nibbles.len() + 1) / 2
                let packed = nibbles.pack();
                let expected_packed_len = nibbles.len().div_ceil(2);
                prop_assert_eq!(packed.len(), expected_packed_len,
                    "packed length should be (nibbles.len() + 1) / 2");

                // Test unpack() length consistency: unpacked.len() == bytes.len() * 2
                let unpacked = Nibbles::unpack(&packed);
                prop_assert_eq!(unpacked.len(), packed.len() * 2,
                    "unpacked length should be bytes.len() * 2");

                // Test to_vec() length consistency
                let vec_result = nibbles.to_vec();
                prop_assert_eq!(vec_result.len(), nibbles.len(),
                    "to_vec() result should have same length as original");

                // Test join() length consistency
                let joined = nibbles.join(&other);
                prop_assert_eq!(joined.len(), nibbles.len() + other.len(),
                    "joined length should be sum of both lengths");

                // Test slice() length consistency
                let valid_start = slice_start.min(nibbles.len());
                let valid_end = slice_end.max(valid_start).min(nibbles.len());
                let sliced = nibbles.slice(valid_start..valid_end);
                prop_assert_eq!(sliced.len(), valid_end - valid_start,
                    "sliced length should be end - start");

                // Test common_prefix_length bounds
                let common_len = nibbles.common_prefix_length(other.as_slice());
                prop_assert!(common_len <= nibbles.len(),
                    "common_prefix_length should not exceed first sequence length");
                prop_assert!(common_len <= other.len(),
                    "common_prefix_length should not exceed second sequence length");

                // Test iterator consistency
                let iter_count = nibbles.iter().count();
                prop_assert_eq!(iter_count, nibbles.len(),
                    "iterator count should match len()");

                // Test first() and last() consistency with length
                if nibbles.is_empty() {
                    prop_assert!(nibbles.first().is_none(),
                        "first() should be None for empty nibbles");
                    prop_assert!(nibbles.last().is_none(),
                        "last() should be None for empty nibbles");
                } else {
                    prop_assert!(nibbles.first().is_some(),
                        "first() should be Some for non-empty nibbles");
                    prop_assert!(nibbles.last().is_some(),
                        "last() should be Some for non-empty nibbles");
                    prop_assert_eq!(nibbles.first(), Some(nibbles[0]),
                        "first() should match nibbles[0]");
                    prop_assert_eq!(nibbles.last(), Some(nibbles[nibbles.len() - 1]),
                        "last() should match nibbles[len-1]");
                }

                // Test get() bounds consistency
                for i in 0..nibbles.len() {
                    prop_assert!(nibbles.get(i).is_some(),
                        "get({}) should be Some for valid index", i);
                }
                prop_assert!(nibbles.get(nibbles.len()).is_none(),
                    "get(len) should be None");
            }

            // ========== EDGE CASE AND INVARIANT PROPERTY TESTS ==========

            /// Test fundamental invariant: all nibbles are always ≤ 0xF
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn nibbles_invariant_always_valid(nibbles in any::<Nibbles>()) {
                for (i, &nibble) in nibbles.iter().enumerate() {
                    prop_assert!(nibble <= 0xF,
                        "Invariant violation: nibble at index {} has value 0x{:X} > 0xF", i, nibble);
                }
                prop_assert!(valid_nibbles(nibbles.as_slice()),
                    "valid_nibbles() helper should confirm all nibbles are valid");
            }

            /// Test length limits and capacity behavior
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn capacity_limits_behavior(nibbles_vec in vec(0u8..=0xF, 0..=64)) {
                if nibbles_vec.len() <= 64 {
                    // Should succeed within capacity
                    let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                    prop_assert_eq!(nibbles.len(), nibbles_vec.len());
                    prop_assert!(nibbles.len() <= 64, "Length should not exceed 64");
                }
            }

            /// Test maximum capacity edge case (exactly 64 nibbles)
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn max_capacity_edge_case(nibbles_vec in vec(0u8..=0xF, 64..=64)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                prop_assert_eq!(nibbles.len(), 64);
                prop_assert!(valid_nibbles(nibbles.as_slice()));

                // Test all access patterns at maximum capacity
                prop_assert_eq!(nibbles.first(), Some(nibbles_vec[0]));
                prop_assert_eq!(nibbles.last(), Some(nibbles_vec[63]));
                prop_assert!(nibbles.get(63).is_some());
                prop_assert!(nibbles.get(64).is_none());
            }

            /// Test overflow/underflow behavior in operations
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn overflow_underflow_operations(nibbles in any::<Nibbles>()) {
                // Test get() with overflow values
                prop_assert!(nibbles.get(usize::MAX).is_none(),
                    "get(usize::MAX) should return None");

                // Test get_byte() with overflow values
                prop_assert!(nibbles.get_byte(usize::MAX).is_none(),
                    "get_byte(usize::MAX) should return None");

                // Test slice operations at boundaries
                if !nibbles.is_empty() {
                    let len = nibbles.len();
                    prop_assert!(nibbles.get(len).is_none(),
                        "get(len) should return None");
                    prop_assert!(nibbles.get(len - 1).is_some(),
                        "get(len-1) should return Some");
                }
            }

            /// Test memory safety with boundary conditions
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn memory_safety_boundaries(nibbles in any::<Nibbles>()) {
                let len = nibbles.len();

                // Test safe access patterns
                for i in 0..len {
                    let nibble = nibbles[i];
                    prop_assert!(nibble <= 0xF, "Accessed nibble should be valid");
                    prop_assert_eq!(nibbles.get(i), Some(&nibble),
                        "get() should match indexing");
                }

                // Test boundary access
                if len > 0 {
                    prop_assert!(nibbles.get(len - 1).is_some());
                }
                prop_assert!(nibbles.get(len).is_none());

                // Test pack/unpack safety at boundaries
                if len % 2 == 0 && len > 0 {
                    let packed = nibbles.pack();
                    prop_assert_eq!(packed.len(), len / 2);
                    let unpacked = Nibbles::unpack(&packed);
                    prop_assert_eq!(unpacked.len(), len);
                }
            }

            /// Test consistency between checked and unchecked variants
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn checked_unchecked_consistency(valid_nibbles in vec(0u8..=0xF, 0..64)) {
                let checked = Nibbles::from_nibbles(&valid_nibbles);
                let unchecked = Nibbles::from_nibbles_unchecked(&valid_nibbles);

                prop_assert_eq!(checked.len(), unchecked.len(),
                    "Checked and unchecked variants should have same length");
                prop_assert_eq!(checked.as_slice(), unchecked.as_slice(),
                    "Checked and unchecked variants should have same content");

                // Test with Vec variants too
                let checked_vec = Nibbles::from_vec(valid_nibbles.clone());
                let unchecked_vec = Nibbles::from_vec_unchecked(valid_nibbles);

                prop_assert_eq!(checked_vec.len(), unchecked_vec.len());
                prop_assert_eq!(checked_vec.as_slice(), unchecked_vec.as_slice());
            }

            /// Test invalid input handling across all methods
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            #[should_panic(expected = "attempted to create invalid nibbles")]
            fn invalid_input_from_nibbles_panics(
                valid_part in vec(0u8..=0xF, 0..30),
                invalid_nibble in 0x10u8..=0xFF,
                insert_pos in 0..=30usize
            ) {
                let mut nibbles_vec = valid_part;
                let pos = insert_pos.min(nibbles_vec.len());
                nibbles_vec.insert(pos, invalid_nibble);

                // Should panic due to invalid nibble
                let _ = Nibbles::from_nibbles(&nibbles_vec);
            }

            /// Test invalid input handling for from_vec
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            #[should_panic(expected = "attempted to create invalid nibbles")]
            fn invalid_input_from_vec_panics(
                valid_part in vec(0u8..=0xF, 0..30),
                invalid_nibble in 0x10u8..=0xFF,
                insert_pos in 0..=30usize
            ) {
                let mut nibbles_vec = valid_part;
                let pos = insert_pos.min(nibbles_vec.len());
                nibbles_vec.insert(pos, invalid_nibble);

                // Should panic due to invalid nibble
                let _ = Nibbles::from_vec(nibbles_vec);
            }

            /// Test edge cases with maximum length sequences
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn max_length_sequence_operations(pattern in 0u8..=0xF) {
                let max_nibbles = vec![pattern; 64];
                let nibbles = Nibbles::from_nibbles(&max_nibbles);

                prop_assert_eq!(nibbles.len(), 64);
                prop_assert!(nibbles.iter().all(|&n| n == pattern));

                // Test operations on max-length sequence
                prop_assert_eq!(nibbles.first(), Some(pattern));
                prop_assert_eq!(nibbles.last(), Some(pattern));

                // Test packing/unpacking with max length (even case)
                let packed = nibbles.pack();
                prop_assert_eq!(packed.len(), 32);
                let unpacked = Nibbles::unpack(&packed);
                prop_assert_eq!(unpacked.len(), 64);
                prop_assert_eq!(unpacked.as_slice(), nibbles.as_slice());
            }

            /// Stress test boundary conditions with various operations
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn stress_test_boundary_conditions(
                nibbles in any::<Nibbles>(),
                access_indices in vec(any::<usize>(), 0..20)
            ) {
                let len = nibbles.len();

                // Test multiple boundary accesses
                for &idx in &access_indices {
                    if idx < len {
                        prop_assert!(nibbles.get(idx).is_some());
                        prop_assert!(nibbles[idx] <= 0xF);
                    } else {
                        prop_assert!(nibbles.get(idx).is_none());
                    }
                }

                // Test byte access boundaries
                let byte_len = len.div_ceil(2);
                for &idx in &access_indices {
                    if idx < byte_len {
                        prop_assert!(nibbles.get_byte(idx).is_some());
                    } else {
                        prop_assert!(nibbles.get_byte(idx).is_none());
                    }
                }

                // Test slicing operations
                if len > 0 {
                    let mid = len / 2;
                    let first_half = &nibbles.as_slice()[..mid];
                    let second_half = &nibbles.as_slice()[mid..];

                    prop_assert!(valid_nibbles(first_half));
                    prop_assert!(valid_nibbles(second_half));
                    prop_assert_eq!(first_half.len() + second_half.len(), len);
                }
            }

            /// Test pack/unpack invariants with edge cases
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn pack_unpack_invariants_edge_cases(bytes in vec(any::<u8>(), 0..32)) {
                let nibbles = Nibbles::unpack(&bytes);

                // Fundamental invariants
                prop_assert_eq!(nibbles.len(), bytes.len() * 2);
                prop_assert!(valid_nibbles(nibbles.as_slice()));

                // Roundtrip invariant
                let packed = nibbles.pack();
                prop_assert_eq!(packed.as_slice(), &bytes[..]);

                // Each nibble should be derived correctly from original bytes
                for (byte_idx, &byte) in bytes.iter().enumerate() {
                    let expected_high = byte >> 4;
                    let expected_low = byte & 0x0F;

                    prop_assert_eq!(nibbles[byte_idx * 2], expected_high);
                    prop_assert_eq!(nibbles[byte_idx * 2 + 1], expected_low);

                    // Verify both nibbles are valid
                    prop_assert!(expected_high <= 0xF);
                    prop_assert!(expected_low <= 0xF);
                }
            }

            /// Test increment operation edge cases and invariants
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn increment_operation_invariants(nibbles in any::<Nibbles>()) {
                let original_len = nibbles.len();

                if let Some(incremented) = nibbles.increment() {
                    // Increment should preserve length and validity
                    prop_assert_eq!(incremented.len(), original_len);
                    prop_assert!(valid_nibbles(incremented.as_slice()));

                    // Should be lexicographically greater (unless overflow)
                    let original_slice = nibbles.as_slice();
                    let incremented_slice = incremented.as_slice();

                    prop_assert!(incremented_slice >= original_slice ||
                        original_slice.iter().all(|&n| n == 0xF),
                        "Increment should produce larger value unless overflow");
                }

                // Test specific overflow case: all 0xF should return None
                if nibbles.iter().all(|&n| n == 0xF) && !nibbles.is_empty() {
                    prop_assert!(nibbles.increment().is_none(),
                        "Incrementing all-0xF sequence should overflow to None");
                }
            }

            /// Test increment produces lexicographically next sequence
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn increment_lexicographic_ordering(nibbles_vec in vec(0u8..=0xF, 0..50)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                
                if let Some(incremented) = nibbles.increment() {
                    let original_slice = nibbles.as_slice();
                    let incremented_slice = incremented.as_slice();
                    
                    // Should be lexicographically greater
                    prop_assert!(incremented_slice > original_slice,
                        "Incremented nibbles should be lexicographically greater than original");
                    
                    // Should be the immediate next sequence - no valid sequence should exist between them
                    let mut test_vec = nibbles_vec.clone();
                    let mut found_increment = false;
                    
                    // Try to find a sequence that would be between original and incremented
                    for i in (0..test_vec.len()).rev() {
                        if test_vec[i] < 0xF {
                            test_vec[i] += 1;
                            // Reset all subsequent nibbles to 0
                            for item in test_vec.iter_mut().skip(i + 1) {
                                *item = 0;
                            }
                            found_increment = true;
                            break;
                        }
                    }
                    
                    if found_increment {
                        prop_assert_eq!(&test_vec[..], incremented_slice,
                            "Increment should produce the immediate next lexicographic sequence");
                    }
                }
            }

            /// Test increment overflow behavior with edge cases
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn increment_overflow_behavior(nibbles_vec in vec(0u8..=0xF, 1..20)) {
                let _nibbles = Nibbles::from_nibbles(&nibbles_vec);
                
                // Test that all-0xF sequences overflow to None
                let all_f_nibbles = Nibbles::from_nibbles(vec![0xF; nibbles_vec.len()]);
                prop_assert!(all_f_nibbles.increment().is_none(),
                    "All-0xF sequence should overflow to None");
                
                // Test sequences that are one increment away from overflow
                if nibbles_vec.len() > 1 {
                    let mut almost_overflow = vec![0xF; nibbles_vec.len()];
                    // Make last nibble 0xE instead of 0xF (since increment works right-to-left)
                    let last_idx = almost_overflow.len() - 1;
                    almost_overflow[last_idx] = 0xE;
                    let almost_overflow_nibbles = Nibbles::from_nibbles(&almost_overflow);
                    
                    if let Some(incremented) = almost_overflow_nibbles.increment() {
                        // Should become all 0xF
                        prop_assert!(incremented.iter().all(|&n| n == 0xF),
                            "Incrementing near-overflow should produce all-0xF sequence");
                        
                        // And incrementing that should overflow
                        prop_assert!(incremented.increment().is_none(),
                            "Incrementing result of near-overflow should overflow to None");
                    }
                }
                
                // Test carry propagation
                if nibbles_vec.len() > 2 {
                    let mut carry_test = vec![0x5; nibbles_vec.len()];
                    // Set last few nibbles to 0xF to test carry propagation
                    for i in (carry_test.len().saturating_sub(3))..carry_test.len() {
                        carry_test[i] = 0xF;
                    }
                    let carry_nibbles = Nibbles::from_nibbles(&carry_test);
                    
                    if let Some(incremented) = carry_nibbles.increment() {
                        // The rightmost non-0xF nibble should be incremented
                        // and all nibbles to its right should become 0
                        let expected_increment_pos = carry_test.len().saturating_sub(4);
                        if expected_increment_pos < carry_test.len() {
                            prop_assert_eq!(incremented[expected_increment_pos], 0x6,
                                "Carry should increment the rightmost non-0xF nibble");
                            
                            // All nibbles after that should be 0
                            for i in (expected_increment_pos + 1)..incremented.len() {
                                prop_assert_eq!(incremented[i], 0,
                                    "Nibbles after carry position should be reset to 0");
                            }
                        }
                    }
                }
            }

            /// Test increment correctness with specific patterns
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn increment_correctness_patterns(
                prefix_len in 0usize..10,
                suffix_len in 0usize..10,
                middle_nibble in 0u8..=0xE // Not 0xF to ensure incrementable
            ) {
                if prefix_len + suffix_len < 50 { // Keep total length reasonable
                    let mut nibbles_vec = vec![0x0; prefix_len];
                    nibbles_vec.push(middle_nibble);
                    nibbles_vec.extend(vec![0xF; suffix_len]);
                    
                    let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                    
                    if let Some(incremented) = nibbles.increment() {
                        // The middle nibble should be incremented
                        prop_assert_eq!(incremented[prefix_len], middle_nibble + 1,
                            "Middle nibble should be incremented from {:x} to {:x}",
                            middle_nibble, middle_nibble + 1);
                        
                        // Prefix should remain unchanged
                        for i in 0..prefix_len {
                            prop_assert_eq!(incremented[i], 0x0,
                                "Prefix nibbles should remain unchanged");
                        }
                        
                        // Suffix should become all zeros due to carry
                        for i in (prefix_len + 1)..incremented.len() {
                            prop_assert_eq!(incremented[i], 0x0,
                                "Suffix nibbles should be reset to 0 after carry");
                        }
                    }
                }
            }

            /// Test is_leaf detection based on last element being 16
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn is_leaf_detection(nibbles_vec in vec(0u8..=0xF, 0..50)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                
                // For regular nibbles (all elements 0-15), should not be leaf
                prop_assert!(!nibbles.is_leaf(),
                    "Regular nibbles with all elements 0-15 should not be leaf");
                
                // Test with explicit leaf marker (16)
                if !nibbles_vec.is_empty() {
                    let mut leaf_vec = nibbles_vec.clone();
                    leaf_vec.push(16);
                    let leaf_nibbles = Nibbles::from_nibbles_unchecked(&leaf_vec);
                    
                    prop_assert!(leaf_nibbles.is_leaf(),
                        "Nibbles ending with 16 should be identified as leaf");
                    
                    // Test that removing the leaf marker makes it non-leaf
                    let mut non_leaf_vec = leaf_vec.clone();
                    non_leaf_vec.pop();
                    let non_leaf_nibbles = Nibbles::from_nibbles(&non_leaf_vec);
                    
                    prop_assert!(!non_leaf_nibbles.is_leaf(),
                        "Nibbles without 16 at end should not be leaf");
                }
                
                // Test edge case: empty nibbles
                let empty_nibbles = Nibbles::new();
                prop_assert!(!empty_nibbles.is_leaf(),
                    "Empty nibbles should not be leaf");
                
                // Test single element cases
                let single_leaf = Nibbles::from_nibbles_unchecked([16]);
                prop_assert!(single_leaf.is_leaf(),
                    "Single element [16] should be leaf");
                
                for nibble in 0u8..=15u8 {
                    let single_non_leaf = Nibbles::from_nibbles([nibble]);
                    prop_assert!(!single_non_leaf.is_leaf(),
                        "Single element [{}] should not be leaf", nibble);
                }
            }

            /// Test is_leaf with various trailing patterns
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn is_leaf_trailing_patterns(
                prefix in vec(0u8..=0xF, 0..20),
                trailing_value in 0u8..=20u8
            ) {
                if !prefix.is_empty() || trailing_value == 16 {
                    let mut nibbles_vec = prefix;
                    nibbles_vec.push(trailing_value);
                    
                    let nibbles = if trailing_value <= 15 {
                        Nibbles::from_nibbles(&nibbles_vec)
                    } else {
                        Nibbles::from_nibbles_unchecked(&nibbles_vec)
                    };
                    
                    let expected_is_leaf = trailing_value == 16;
                    prop_assert_eq!(nibbles.is_leaf(), expected_is_leaf,
                        "Nibbles ending with {} should {} be leaf",
                        trailing_value, if expected_is_leaf { "" } else { "not" });
                }
            }

            /// Test integration of increment with is_leaf
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn increment_is_leaf_integration(nibbles_vec in vec(0u8..=0xF, 0..20)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                
                // Regular nibbles should not be leaf
                prop_assert!(!nibbles.is_leaf(),
                    "Regular nibbles should not be leaf before increment");
                
                if let Some(incremented) = nibbles.increment() {
                    // Incremented regular nibbles should still not be leaf
                    prop_assert!(!incremented.is_leaf(),
                        "Incremented regular nibbles should not be leaf");
                    
                    // Length should be preserved
                    prop_assert_eq!(incremented.len(), nibbles.len(),
                        "Increment should preserve length");
                }
                
                // Test with leaf nibbles (ending in 16)
                // NOTE: We cannot test increment() on leaf nibbles because increment() has
                // a debug assertion that all nibbles must be <= 0xF, but leaf nibbles contain 16.
                // This is by design - increment() is only meant for valid nibble sequences.
                if !nibbles_vec.is_empty() {
                    let mut leaf_vec = nibbles_vec.clone();
                    leaf_vec.push(16);
                    let leaf_nibbles = Nibbles::from_nibbles_unchecked(&leaf_vec);
                    
                    prop_assert!(leaf_nibbles.is_leaf(),
                        "Nibbles with leaf marker should be leaf");
                    
                    // Test that leaf nibbles have the expected structure
                    prop_assert_eq!(leaf_nibbles.len(), nibbles_vec.len() + 1,
                        "Leaf nibbles should have one extra element");
                    prop_assert_eq!(leaf_nibbles.last(), Some(16),
                        "Leaf nibbles should end with 16");
                    
                    // Test that prefix without leaf marker is not leaf
                    let prefix_slice = &leaf_nibbles.as_slice()[..nibbles_vec.len()];
                    let prefix_nibbles = Nibbles::from_nibbles(prefix_slice);
                    prop_assert!(!prefix_nibbles.is_leaf(),
                        "Prefix without leaf marker should not be leaf");
                }
            }

            /// Test increment with boundary conditions and edge cases
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn increment_boundary_conditions(length in 1usize..=10) {
                // Test incrementing sequence of all zeros
                let all_zeros = Nibbles::from_nibbles(vec![0; length]);
                if let Some(incremented) = all_zeros.increment() {
                    // Should become [0, 0, ..., 0, 1]
                    prop_assert_eq!(incremented.len(), length, "Length should be preserved");
                    for i in 0..(length - 1) {
                        prop_assert_eq!(incremented[i], 0, "Leading nibbles should remain 0");
                    }
                    prop_assert_eq!(incremented[length - 1], 1, "Last nibble should become 1");
                }
                
                // Test incrementing sequence with only last nibble non-zero
                for last_nibble in 0u8..=0xE {
                    let mut test_vec = vec![0; length];
                    test_vec[length - 1] = last_nibble;
                    let test_nibbles = Nibbles::from_nibbles(&test_vec);
                    
                    if let Some(incremented) = test_nibbles.increment() {
                        prop_assert_eq!(incremented.len(), length, "Length should be preserved");
                        for i in 0..(length - 1) {
                            prop_assert_eq!(incremented[i], 0, "Leading nibbles should remain 0");
                        }
                        prop_assert_eq!(incremented[length - 1], last_nibble + 1,
                            "Last nibble should be incremented from {} to {}", last_nibble, last_nibble + 1);
                    }
                }
                
                // Test single nibble sequences
                for single_nibble in 0u8..=0xE {
                    let single = Nibbles::from_nibbles([single_nibble]);
                    if let Some(incremented) = single.increment() {
                        prop_assert_eq!(incremented.len(), 1, "Single nibble length should be preserved");
                        prop_assert_eq!(incremented[0], single_nibble + 1,
                            "Single nibble should be incremented from {} to {}", single_nibble, single_nibble + 1);
                    }
                }
                
                // Test that single 0xF overflows
                let single_f = Nibbles::from_nibbles([0xF]);
                prop_assert!(single_f.increment().is_none(),
                    "Single 0xF should overflow to None");
            }

            /// Test integration with construction and manipulation methods
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn increment_construction_integration(nibbles_vec in vec(0u8..=0xF, 0..30)) {
                let nibbles = Nibbles::from_nibbles(&nibbles_vec);
                
                if let Some(incremented) = nibbles.increment() {
                    // Test roundtrip with pack/unpack
                    let packed = incremented.pack();
                    let unpacked = Nibbles::unpack(&packed);
                    
                    if nibbles_vec.len() % 2 == 0 {
                        // Even length: perfect roundtrip
                        prop_assert_eq!(unpacked.as_slice(), incremented.as_slice(),
                            "Even-length increment should survive pack/unpack roundtrip");
                    } else {
                        // Odd length: gets zero-padded
                        prop_assert_eq!(&unpacked[..incremented.len()], incremented.as_slice(),
                            "Original incremented nibbles should be preserved after pack/unpack");
                    }
                    
                    // Test that incremented nibbles can be cloned
                    let cloned = incremented.clone();
                    prop_assert_eq!(cloned.as_slice(), incremented.as_slice(),
                        "Cloned incremented nibbles should be identical");
                    
                    // Test that incremented nibbles can be converted to vec
                    let vec_result = incremented.to_vec();
                    prop_assert_eq!(&vec_result[..], incremented.as_slice(),
                        "to_vec() should preserve incremented nibbles");
                    
                    // Test that we can create new nibbles from incremented result
                    let reconstructed = Nibbles::from_nibbles(&vec_result);
                    prop_assert_eq!(reconstructed.as_slice(), incremented.as_slice(),
                        "Reconstructed nibbles should match incremented nibbles");
                }
            }

            /// Test with_capacity invariants
            #[test]
            #[cfg_attr(miri, ignore = "no proptest")]
            fn with_capacity_invariants(capacity in 0usize..=128) {
                let nibbles = Nibbles::with_capacity(capacity);

                // Should create empty nibbles regardless of capacity
                prop_assert_eq!(nibbles.len(), 0);
                prop_assert!(nibbles.is_empty());
                prop_assert_eq!(nibbles.as_slice(), &[]);

                // Should still maintain nibble validity (vacuously true for empty)
                prop_assert!(valid_nibbles(nibbles.as_slice()));
            }
        }
    }
}
