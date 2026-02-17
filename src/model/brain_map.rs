// =============================================================================
// model/brain_map.rs — Bản Đồ Não Bộ (Brain Region Management)
// =============================================================================
//
// Triết lý: Thay vì load 1 file to (300MB+), ta chia não thành các vùng:
//   - Shallow Reflex: Ngữ pháp, chém gió (~20MB)
//   - Deep Logic: Code, toán, suy luận (~50MB)
//   - Hard Fact: Wikipedia, tra cứu (~100MB)
//
// Chỉ load vùng não được Instinct Core yêu cầu → RAM < 100MB.
//
// Kỹ thuật:
//   - Memory-mapped file: Toàn bộ dữ liệu trong 1 file lớn
//   - Offset-based access: Mỗi vùng có offset & size riêng
//   - Lazy loading: Chỉ truy cập → OS tự động page in
//   - Metadata: File header chứa bảng offset/size của từng vùng
// =============================================================================

use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Loại vùng não (tương ứng với BrainRegion trong instinct.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionType {
    ShallowReflex = 0, // Nhẹ, nhanh (chat, ngữ pháp)
    DeepLogic = 1,     // Nặng, chậm (code, toán)
    HardFact = 2,      // Nặng, chỉ đọc (wiki, tra cứu)
}

impl RegionType {
    pub fn from_id(id: usize) -> Self {
        match id {
            0 => Self::ShallowReflex,
            1 => Self::DeepLogic,
            2 => Self::HardFact,
            _ => Self::ShallowReflex,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ShallowReflex => "Shallow",
            Self::DeepLogic => "Deep",
            Self::HardFact => "Fact",
        }
    }
}

/// Metadata của 1 vùng não
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BrainRegion {
    pub id: usize,
    pub region_type: RegionType,
    /// Offset trong file mmap (bytes)
    pub file_offset: usize,
    /// Kích thước dữ liệu (bytes)
    pub byte_size: usize,
    /// Số layers trong vùng này (có thể khác nhau)
    pub n_layers: usize,
    /// Dimension (embedding size)
    pub dim: usize,
}

impl BrainRegion {
    pub fn new(
        id: usize,
        region_type: RegionType,
        file_offset: usize,
        byte_size: usize,
        n_layers: usize,
        dim: usize,
    ) -> Self {
        Self {
            id,
            region_type,
            file_offset,
            byte_size,
            n_layers,
            dim,
        }
    }

    /// Tính memory footprint (MB)
    pub fn memory_mb(&self) -> f32 {
        self.byte_size as f32 / 1e6
    }
}

/// Quản lý toàn bộ bản đồ não
pub struct BrainMap {
    /// Danh sách các vùng não
    pub regions: Vec<BrainRegion>,
    /// Memory-mapped file chứa tất cả dữ liệu
    pub data_source: Mmap,
    /// Vùng đang được active (cho stats)
    pub active_region: Option<usize>,
}

impl BrainMap {
    /// Load bản đồ não từ file .brain
    /// Format file:
    ///   [Header: 4 bytes magic + 4 bytes n_regions]
    ///   [Region metadata: n_regions × 48 bytes]
    ///   [Region data: concatenated weights]
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open brain map: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap brain: {e}"))?;

        // Parse header
        if mmap.len() < 8 {
            return Err("File too small for header".into());
        }

        let magic = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
        if magic != 0x4252414E {
            // "BRAN"
            return Err(format!("Invalid magic: 0x{:X}", magic));
        }

        let n_regions = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;

        // Parse region metadata
        let mut regions = Vec::new();
        let mut offset = 8;

        for i in 0..n_regions {
            if offset + 48 > mmap.len() {
                return Err("Truncated region metadata".into());
            }

            let region_type_id = u32::from_le_bytes([
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
            ]) as usize;

            let file_offset = u64::from_le_bytes([
                mmap[offset + 4],
                mmap[offset + 5],
                mmap[offset + 6],
                mmap[offset + 7],
                mmap[offset + 8],
                mmap[offset + 9],
                mmap[offset + 10],
                mmap[offset + 11],
            ]) as usize;

            let byte_size = u64::from_le_bytes([
                mmap[offset + 12],
                mmap[offset + 13],
                mmap[offset + 14],
                mmap[offset + 15],
                mmap[offset + 16],
                mmap[offset + 17],
                mmap[offset + 18],
                mmap[offset + 19],
            ]) as usize;

            let n_layers = u32::from_le_bytes([
                mmap[offset + 20],
                mmap[offset + 21],
                mmap[offset + 22],
                mmap[offset + 23],
            ]) as usize;

            let dim = u32::from_le_bytes([
                mmap[offset + 24],
                mmap[offset + 25],
                mmap[offset + 26],
                mmap[offset + 27],
            ]) as usize;

            regions.push(BrainRegion::new(
                i,
                RegionType::from_id(region_type_id),
                file_offset,
                byte_size,
                n_layers,
                dim,
            ));

            offset += 48;
        }

        println!(
            "[brain_map] Loaded {} regions, total size: {:.1} MB",
            regions.len(),
            mmap.len() as f32 / 1e6
        );

        for region in &regions {
            println!(
                "  Region {}: {} ({:.1} MB, {} layers)",
                region.id,
                region.region_type.name(),
                region.memory_mb(),
                region.n_layers
            );
        }

        Ok(Self {
            regions,
            data_source: mmap,
            active_region: None,
        })
    }

    /// Tạo file .brain mới (dummy data cho testing)
    pub fn create_dummy(path: &Path, configs: &[(RegionType, usize, usize)]) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| format!("create brain: {e}"))?;

        // Write header
        file.write_all(&0x4252414E_u32.to_le_bytes()).map_err(|e| format!("write header: {e}"))?; // Magic "BRAN"
        file.write_all(&(configs.len() as u32).to_le_bytes()).map_err(|e| format!("write count: {e}"))?;

        let header_size = 8 + configs.len() * 48;
        let mut current_offset = header_size;
        let mut metadata_buffer = Vec::new();

        // Write region metadata
        for &(region_type, n_layers, dim) in configs.iter() {
            let bytes_per_weight = 4; // Float32
            let params_per_layer = dim * dim * 4; // Simplified: Q, K, V, O matrices
            let byte_size = n_layers * params_per_layer * bytes_per_weight;

            metadata_buffer.extend_from_slice(&(region_type as u32).to_le_bytes());
            metadata_buffer.extend_from_slice(&(current_offset as u64).to_le_bytes());
            metadata_buffer.extend_from_slice(&(byte_size as u64).to_le_bytes());
            metadata_buffer.extend_from_slice(&(n_layers as u32).to_le_bytes());
            metadata_buffer.extend_from_slice(&(dim as u32).to_le_bytes());
            metadata_buffer.resize(metadata_buffer.len() + 20, 0); // Padding to 48 bytes

            current_offset += byte_size;
        }

        file.write_all(&metadata_buffer).map_err(|e| format!("write metadata: {e}"))?;

        // Write dummy region data
        for &(_, n_layers, dim) in configs {
            let params_per_layer = dim * dim * 4;
            let total_params = n_layers * params_per_layer;
            let dummy_data = vec![0u8; total_params * 4]; // All zeros for now
            file.write_all(&dummy_data).map_err(|e| format!("write region data: {e}"))?;
        }

        println!(
            "[brain_map] Created dummy brain map: {:.1} MB",
            current_offset as f32 / 1e6
        );

        Ok(())
    }

    /// Lấy dữ liệu weights của 1 vùng não (zero-copy slice)
    pub fn get_weights(&mut self, region_id: usize) -> Option<&[u8]> {
        if region_id >= self.regions.len() {
            return None;
        }

        let region = &self.regions[region_id];
        self.active_region = Some(region_id);

        let start = region.file_offset;
        let end = start + region.byte_size;

        if end > self.data_source.len() {
            return None;
        }

        Some(&self.data_source[start..end])
    }

    /// Lấy metadata của vùng não
    pub fn get_region(&self, region_id: usize) -> Option<&BrainRegion> {
        self.regions.get(region_id)
    }

    /// Tìm vùng não theo loại
    pub fn find_region_by_type(&self, region_type: RegionType) -> Option<&BrainRegion> {
        self.regions.iter().find(|r| r.region_type == region_type)
    }

    /// Tính tổng RAM đã dùng (ước lượng - OS page-in)
    pub fn estimated_ram_usage_mb(&self) -> f32 {
        // Chỉ vùng active thực sự trong RAM
        match self.active_region {
            Some(id) => self.regions[id].memory_mb(),
            None => 0.0,
        }
    }

    /// Stats: Tổng số vùng
    pub fn n_regions(&self) -> usize {
        self.regions.len()
    }
    
    /// Activate a specific brain region for inference
    pub fn activate_region(&mut self, region_id: usize) {
        if region_id < self.regions.len() {
            self.active_region = Some(region_id);
        }
    }
}

// =============================================================================
// Helper: Phân tích weights từ raw bytes
// =============================================================================

/// Cast byte slice thành Float32 slice (unsafe nhưng zero-copy)
pub fn bytes_as_f32_slice(bytes: &[u8]) -> &[f32] {
    let ptr = bytes.as_ptr() as *const f32;
    let len = bytes.len() / 4;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Trích xuất 1 matrix từ vùng não
/// offset: vị trí bắt đầu (số Float32)
/// rows, cols: kích thước matrix
pub fn extract_matrix<'a>(
    region_weights: &'a [u8],
    offset: usize,
    rows: usize,
    cols: usize,
) -> Option<&'a [f32]> {
    let floats = bytes_as_f32_slice(region_weights);
    let start = offset;
    let end = start + rows * cols;

    if end > floats.len() {
        return None;
    }

    Some(&floats[start..end])
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_brain_map() {
        let path = Path::new("test_brain.brain");
        let configs = vec![
            (RegionType::ShallowReflex, 4, 256), // 4 layers, dim=256
            (RegionType::DeepLogic, 8, 512),     // 8 layers, dim=512
        ];

        BrainMap::create_dummy(path, &configs).unwrap();
        assert!(path.exists());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_brain_map() {
        let path = Path::new("test_brain2.brain");
        let configs = vec![(RegionType::ShallowReflex, 2, 128)];

        BrainMap::create_dummy(path, &configs).unwrap();

        let mut brain = BrainMap::load(path).unwrap();
        assert_eq!(brain.n_regions(), 1);

        let weights = brain.get_weights(0);
        assert!(weights.is_some());

        let region = brain.get_region(0).unwrap();
        assert_eq!(region.region_type, RegionType::ShallowReflex);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_find_region_by_type() {
        let path = Path::new("test_brain3.brain");
        let configs = vec![
            (RegionType::ShallowReflex, 2, 128),
            (RegionType::DeepLogic, 4, 256),
        ];

        BrainMap::create_dummy(path, &configs).unwrap();
        let brain = BrainMap::load(path).unwrap();

        let deep = brain.find_region_by_type(RegionType::DeepLogic);
        assert!(deep.is_some());
        assert_eq!(deep.unwrap().n_layers, 4);

        std::fs::remove_file(path).ok();
    }
}
