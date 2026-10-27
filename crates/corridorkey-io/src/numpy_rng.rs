//! `np.random.default_rng(seed).integers(low, high)`, bit for bit: the product picks the
//! `random_middle` start frame with it (`probe_video_meta`). NumPy's `SeedSequence` (pool of four
//! 32-bit words), `PCG64` (128-bit LCG, XSL-RR output, 32-bit draws buffered from 64-bit ones) and
//! the Lemire bounded draw of `_bounded_integers` (ranges below 2^32: one 32-bit draw per try).

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const POOL_SIZE: usize = 4;

/// `SeedSequence(seed).generate_state(4, np.uint64)`.
fn seed_sequence_state(seed: u64) -> [u64; 4] {
    // `_int_to_uint32_array`: little-endian 32-bit words, `[0]` for 0
    let entropy: Vec<u32> = if seed == 0 {
        vec![0]
    } else {
        let mut words = vec![];
        let mut n = seed;
        while n > 0 {
            words.push(n as u32);
            n >>= 32;
        }
        words
    };
    let mut hash_const = INIT_A;
    let mut hashmix = |value: u32| {
        let mut v = value ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; POOL_SIZE];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for src in 0..POOL_SIZE {
        for dst in 0..POOL_SIZE {
            if src != dst {
                let h = hashmix(pool[src]);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    for &e in entropy.iter().skip(POOL_SIZE) {
        for dst in 0..POOL_SIZE {
            let h = hashmix(e);
            pool[dst] = mix(pool[dst], h);
        }
    }
    // generate_state: 8 words cycled from the pool, read as 4 little-endian u64
    let mut hash_const = INIT_B;
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        let mut v = pool[i % POOL_SIZE] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hash_const);
        *w = v ^ (v >> 16);
    }
    std::array::from_fn(|i| words[2 * i] as u64 | (words[2 * i + 1] as u64) << 32)
}

const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

/// NumPy's `PCG64` bit generator.
pub struct Pcg64 {
    state: u128,
    inc: u128,
    /// The high half of the last 64-bit draw, not yet handed out as a 32-bit one.
    buffered: Option<u32>,
}

impl Pcg64 {
    /// `PCG64(SeedSequence(seed))`, as `np.random.default_rng(seed)` builds it.
    pub fn from_seed(seed: u64) -> Self {
        let v = seed_sequence_state(seed);
        let initstate = (v[0] as u128) << 64 | v[1] as u128;
        let initseq = (v[2] as u128) << 64 | v[3] as u128;
        // pcg_setseq_128_srandom_r
        let mut rng = Self { state: 0, inc: (initseq << 1) | 1, buffered: None };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    pub fn next_u32(&mut self) -> u32 {
        if let Some(high) = self.buffered.take() {
            return high;
        }
        let next = self.next_u64();
        self.buffered = Some((next >> 32) as u32);
        next as u32
    }

    /// `Generator.integers(low, high)` (high exclusive, default int64) for `high - low ≤ 2^32`.
    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        assert!(high > low && (high - low - 1) as u64 <= u32::MAX as u64, "range [{low}, {high}) not supported");
        let rng = (high - low - 1) as u32;
        if rng == 0 {
            return low;
        }
        if rng == u32::MAX {
            return low + self.next_u32() as i64;
        }
        // buffered_bounded_lemire_uint32
        let excl = rng + 1;
        let mut m = self.next_u32() as u64 * excl as u64;
        let mut leftover = m as u32;
        if leftover < excl {
            let threshold = (u32::MAX - rng) % excl;
            while leftover < threshold {
                m = self.next_u32() as u64 * excl as u64;
                leftover = m as u32;
            }
        }
        low + (m >> 32) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// From NumPy 2.x: `[np.random.default_rng(s).integers(lo, hi) for ...]`.
    #[test]
    fn matches_numpy() {
        for &(seed, lo, hi, want) in CASES {
            assert_eq!(Pcg64::from_seed(seed).integers(lo, hi), want, "seed {seed}, [{lo}, {hi})");
        }
    }

    #[rustfmt::skip]
    const CASES: &[(u64, i64, i64, i64)] = &[
    (0, 0, 1, 0),
    (0, 5, 6, 5),
    (0, 0, 2, 1),
    (0, 3, 10, 8),
    (0, 25, 76, 68),
    (0, 0, 1000, 850),
    (0, 100, 100001, 85078),
    (0, 0, 2147483648, 1826701615),
    (0, 0, 4294967295, 3653403230),
    (0, 0, 4294967296, 3653403231),
    (1, 0, 1, 0),
    (1, 5, 6, 5),
    (1, 0, 2, 0),
    (1, 3, 10, 6),
    (1, 25, 76, 49),
    (1, 0, 1000, 473),
    (1, 100, 100001, 47372),
    (1, 0, 2147483648, 1016164991),
    (1, 0, 4294967295, 2032329982),
    (1, 0, 4294967296, 2032329983),
    (2, 0, 1, 0),
    (2, 5, 6, 5),
    (2, 0, 2, 1),
    (2, 3, 10, 8),
    (2, 25, 76, 67),
    (2, 0, 1000, 837),
    (2, 100, 100001, 83774),
    (2, 0, 2147483648, 1798679648),
    (2, 0, 4294967295, 3597359296),
    (2, 0, 4294967296, 3597359297),
    (7, 0, 1, 0),
    (7, 5, 6, 5),
    (7, 0, 2, 1),
    (7, 3, 10, 9),
    (7, 25, 76, 73),
    (7, 0, 1000, 944),
    (7, 100, 100001, 94496),
    (7, 0, 2147483648, 2029167941),
    (7, 0, 4294967295, 4058335882),
    (7, 0, 4294967296, 4058335883),
    (42, 0, 1, 0),
    (42, 5, 6, 5),
    (42, 0, 2, 0),
    (42, 3, 10, 3),
    (42, 25, 76, 29),
    (42, 0, 1000, 89),
    (42, 100, 100001, 9016),
    (42, 0, 2147483648, 191664964),
    (42, 0, 4294967295, 383329927),
    (42, 0, 4294967296, 383329928),
    (123, 0, 1, 0),
    (123, 5, 6, 5),
    (123, 0, 2, 0),
    (123, 3, 10, 3),
    (123, 25, 76, 25),
    (123, 0, 1000, 15),
    (123, 100, 100001, 1642),
    (123, 0, 2147483648, 33158374),
    (123, 0, 4294967295, 66316748),
    (123, 0, 4294967296, 66316749),
    (2024, 0, 1, 0),
    (2024, 5, 6, 5),
    (2024, 0, 2, 0),
    (2024, 3, 10, 4),
    (2024, 25, 76, 37),
    (2024, 0, 1000, 241),
    (2024, 100, 100001, 24228),
    (2024, 0, 2147483648, 518677876),
    (2024, 0, 4294967295, 1037355751),
    (2024, 0, 4294967296, 1037355752),
    (99999, 0, 1, 0),
    (99999, 5, 6, 5),
    (99999, 0, 2, 0),
    (99999, 3, 10, 6),
    (99999, 25, 76, 47),
    (99999, 0, 1000, 444),
    (99999, 100, 100001, 44514),
    (99999, 0, 2147483648, 954743357),
    (99999, 0, 4294967295, 1909486713),
    (99999, 0, 4294967296, 1909486714),
    (2147483647, 0, 1, 0),
    (2147483647, 5, 6, 5),
    (2147483647, 0, 2, 1),
    (2147483647, 3, 10, 8),
    (2147483647, 25, 76, 66),
    (2147483647, 0, 1000, 815),
    (2147483647, 100, 100001, 81600),
    (2147483647, 0, 2147483648, 1751933749),
    (2147483647, 0, 4294967295, 3503867497),
    (2147483647, 0, 4294967296, 3503867498),
    (4294967301, 0, 1, 0),
    (4294967301, 5, 6, 5),
    (4294967301, 0, 2, 0),
    (4294967301, 3, 10, 3),
    (4294967301, 25, 76, 31),
    (4294967301, 0, 1000, 132),
    (4294967301, 100, 100001, 13317),
    (4294967301, 0, 2147483648, 284122216),
    (4294967301, 0, 4294967295, 568244432),
    (4294967301, 0, 4294967296, 568244433),
    (9223372036854775807, 0, 1, 0),
    (9223372036854775807, 5, 6, 5),
    (9223372036854775807, 0, 2, 0),
    (9223372036854775807, 3, 10, 6),
    (9223372036854775807, 25, 76, 47),
    (9223372036854775807, 0, 1000, 436),
    (9223372036854775807, 100, 100001, 43688),
    (9223372036854775807, 0, 2147483648, 936983178),
    (9223372036854775807, 0, 4294967295, 1873966355),
    (9223372036854775807, 0, 4294967296, 1873966356),
    ];
}
