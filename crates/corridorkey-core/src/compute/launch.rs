use cubecl::prelude::*;

/// Kernel launch geometry for the access patterns used by the CorridorKey kernels.
///
/// Cube sizes come from the runtime ([`CubeDim::new`]: its plane size along x, a plane count set
/// by the runtime's cores or limits along y), never from constants, and cube counts from the
/// problem size within the runtime's grid limits. CubeCL maps cubes and units to whatever executes
/// the kernel (CPU threads, GPU workgroups).
#[derive(Debug, Clone)]
pub struct LaunchGeometry {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
}

/// Column (x) of the unit in a [`LaunchGeometry::pixels`] launch. Columns run along x and continue
/// along z when they need more cubes than one grid dimension holds; kernels bound-check the result.
#[cube]
pub fn column_position() -> u32 {
    (CUBE_POS_Z * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X
}

/// Image row (y) of the unit in a [`LaunchGeometry::pixels`] launch.
#[cube]
pub fn pixel_row_position() -> u32 {
    ABSOLUTE_POS_Y
}

/// Row reduced by the cube in a [`LaunchGeometry::per_row`] launch (uniform across the cube).
#[cube]
pub fn row_position() -> u32 {
    CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X
}

impl LaunchGeometry {
    /// One unit per element, indexed with the linear `ABSOLUTE_POS`. The cubes are spread over the
    /// runtime's cube-count dimensions, so kernels must bound-check `ABSOLUTE_POS`.
    pub fn elementwise(client: &Client, num_elements: usize) -> Self {
        let n = num_elements.max(1);
        let cube_dim = CubeDim::new(client, n);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, n, cube_dim);
        Self { cube_dim, cube_count }
    }

    /// One unit per pixel `(row, column)` of a `[height, width]` image plane:
    /// [`fn@pixel_row_position`] and [`fn@column_position`]. The cube's x size is the runtime's
    /// plane, so a plane walks neighbouring pixels of one row (coalesced reads in a row-major
    /// image).
    ///
    /// # Panics
    /// If the rows or columns need more cubes than the runtime's whole grid holds.
    pub fn pixels(client: &Client, height: usize, width: usize) -> Self {
        let cube_dim = CubeDim::new(client, height.max(1) * width.max(1));
        let (x, z) = Self::spill(client, width, cube_dim.x);
        let y = (height.max(1) as u32).div_ceil(cube_dim.y);
        assert!(
            y <= client.properties().hardware.max_cube_count.1,
            "{height} rows exceed the runtime's cube grid"
        );
        Self { cube_dim, cube_count: CubeCount::Static(x, y, z) }
    }

    /// One cube per row of a `[rows, cols]` buffer, for cube-wide reductions: [`fn@row_position`] is the
    /// row (cubes along x, continuing along y past the grid's x limit) and `UNIT_POS_X` strides along
    /// it. The cube is the runtime's cube for `cols` units flattened onto x and rounded down to a
    /// power of two, so tree reductions can halve it; kernels size shared memory with
    /// `cube_dim.x` (passed as a comptime value). Kernels bound-check the row.
    ///
    /// # Panics
    /// If the rows need more cubes than the runtime's whole grid holds.
    pub fn per_row(client: &Client, rows: usize, cols: usize) -> Self {
        let units = Self::flat(client, cols).x;
        let cube_dim = CubeDim::new_1d(1 << (u32::BITS - 1 - units.max(1).leading_zeros()));
        let (x, y) = Self::spill(client, rows, 1);
        assert!(y <= client.properties().hardware.max_cube_count.1, "{rows} rows exceed the runtime's cube grid");
        Self { cube_dim, cube_count: CubeCount::Static(x, y, 1) }
    }

    /// One unit per point of `points`, in 1-D cubes of a power-of-two number of units (the
    /// runtime's cube flattened onto x, shrunk until `shared_bytes_per_unit` per unit fits the
    /// device's shared memory) for kernels that share tiles of points in shared memory: the point
    /// is `row_position() * CUBE_DIM_X + UNIT_POS_X`, and kernels size shared memory with
    /// `cube_dim.x` (passed as a comptime value). Every unit of a cube takes part in the tile loads,
    /// so kernels bound-check the point only around their own work.
    ///
    /// # Panics
    /// If one unit's shared memory exceeds the device's, or the points need more cubes than the
    /// runtime's whole grid holds.
    pub fn tiles(client: &Client, points: usize, shared_bytes_per_unit: usize) -> Self {
        let shared = client.properties().hardware.max_shared_memory_size;
        let fit = shared / shared_bytes_per_unit.max(1);
        assert!(fit >= 1, "{shared_bytes_per_unit} bytes of shared memory per unit exceed the device's {shared}");
        let units = Self::flat(client, points).x.min(u32::try_from(fit).unwrap_or(u32::MAX));
        let units = 1 << (u32::BITS - 1 - units.max(1).leading_zeros());
        let (x, y) = Self::spill(client, points, units);
        assert!(y <= client.properties().hardware.max_cube_count.1, "{points} points exceed the runtime's cube grid");
        Self { cube_dim: CubeDim::new_1d(units), cube_count: CubeCount::Static(x, y, 1) }
    }

    /// Units that execute in lock-step (the runtime's plane / warp / subgroup size; 1 where every
    /// unit runs on its own, as on the CPU runtime). It is the x size of
    /// [`Self::pixels`] cubes, so consecutive `column_position`s of a plane share a cube.
    pub fn plane_lanes(client: &Client) -> u32 {
        client.properties().hardware.plane_size_max.max(1)
    }

    /// The runtime's cube for `work` units, flattened onto x (for kernels indexed along x only).
    fn flat(client: &Client, work: usize) -> CubeDim {
        CubeDim::new_1d(CubeDim::new(client, work.max(1)).num_elems())
    }

    /// Cube counts `(x, overflow)` covering `units` along x with `per_cube` units per cube; the
    /// overflow count continues past the grid's x limit (along z for columns, y for rows).
    fn spill(client: &Client, units: usize, per_cube: u32) -> (u32, u32) {
        let max = client.properties().hardware.max_cube_count;
        let cubes = (units.max(1) as u64).div_ceil(per_cube as u64);
        let x = cubes.min(max.0 as u64);
        let overflow = cubes.div_ceil(x);
        assert!(overflow <= max.2.min(max.1) as u64, "{units} units exceed the runtime's cube grid");
        (x as u32, overflow as u32)
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    use cubecl::device::WgpuDevice;

    fn cubes(count: &CubeCount) -> u64 {
        match count {
            CubeCount::Static(x, y, z) => *x as u64 * *y as u64 * *z as u64,
            _ => unreachable!(),
        }
    }

    #[test]
    fn geometries_cover_their_ranges_within_limits() {
        let client = crate::compute::open_device(cubecl::Device::Wgpu(WgpuDevice::default()));
        let max = client.properties().hardware.max_cube_count;
        let within = |c: &CubeCount| matches!(c, CubeCount::Static(x, y, z) if *x <= max.0 && *y <= max.1 && *z <= max.2);

        // 64 frames of 4K RGB: more cubes than one dimension allows.
        let n = 64 * 2160 * 3840 * 3;
        let geom = LaunchGeometry::elementwise(&client, n);
        assert!(within(&geom.cube_count) && cubes(&geom.cube_count) * geom.cube_dim.num_elems() as u64 >= n as u64);

        // Odd sizes; widths beyond one grid dimension spill into z.
        for width in [3_841usize, 5_000_000] {
            let geom = LaunchGeometry::pixels(&client, 2_161, width);
            let CubeCount::Static(x, y, z) = geom.cube_count else { unreachable!() };
            assert!(within(&geom.cube_count));
            assert!(x as u64 * z as u64 * geom.cube_dim.x as u64 >= width as u64 && y * geom.cube_dim.y >= 2_161);
            assert_eq!(geom.cube_dim.x, LaunchGeometry::plane_lanes(&client));
        }
    }
}
