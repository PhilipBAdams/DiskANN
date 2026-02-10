/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use core::fmt::Debug;

use diskann::{ANNError, ANNResult};
use diskann_quantization::product::TransposedTable;

use crate::model::{FixedChunkPQTable, PQCompressedData};

/// The PQ table used during search preprocessing.
///
/// We always use the [`TransposedTable`] for the fast `process_into` distance table
/// computation, and carry the centroid and optional OPQ rotation matrix alongside it
/// so that query preprocessing (centroid subtraction + OPQ rotation) can be applied
/// before the distance table is built.
#[derive(Debug)]
pub struct PQTable {
    table: TransposedTable,
    /// Per-dimension centroid to subtract from the query. All zeros when the dataset
    /// was built without centroid removal.
    centroids: Box<[f32]>,
    /// Optional OPQ rotation matrix (dim × dim, row-major).
    opq_rotation_matrix: Option<Box<[f32]>>,
    /// Dimensionality of the full-precision vectors.
    dim: usize,
}

impl PQTable {
    /// Return a reference to the underlying transposed table.
    pub fn transposed_table(&self) -> &TransposedTable {
        &self.table
    }

    /// Return the per-dimension centroids.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Return the optional OPQ rotation matrix.
    pub fn opq_rotation_matrix(&self) -> Option<&[f32]> {
        self.opq_rotation_matrix.as_deref()
    }

    /// Return the dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the number of PQ chunks.
    pub fn nchunks(&self) -> usize {
        self.table.nchunks()
    }

    /// Return the number of centroids per chunk.
    pub fn ncenters(&self) -> usize {
        self.table.ncenters()
    }
}

#[derive(Debug)]
pub struct PQData {
    // pq pivot table.
    pq_pivot_table: PQTable,

    // pq compressed vectors.
    pq_compressed_data: PQCompressedData,
}

impl PQData {
    pub fn new(
        pq_pivot_table: FixedChunkPQTable,
        pq_compressed_data: PQCompressedData,
    ) -> ANNResult<Self> {
        let dim = pq_pivot_table.get_dim();
        let centroids = pq_pivot_table.get_centroids().into();
        let opq_rotation_matrix = pq_pivot_table.get_opq_rotation_matrix();

        let transposed = TransposedTable::from_parts(
            pq_pivot_table.view_pivots(),
            pq_pivot_table.view_offsets().to_owned(),
        )
        .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?;

        let pq_pivot_table = PQTable {
            table: transposed,
            centroids,
            opq_rotation_matrix,
            dim,
        };

        Ok(Self {
            pq_pivot_table,
            pq_compressed_data,
        })
    }

    /// Get pq_table
    pub fn pq_table(&self) -> &PQTable {
        &self.pq_pivot_table
    }

    /// Return the number of chunks in the underlying PQ schema.
    pub fn get_num_chunks(&self) -> usize {
        self.pq_pivot_table.nchunks()
    }

    /// Return the number of centers in the underlying PQ schema.
    pub fn get_num_centers(&self) -> usize {
        self.pq_pivot_table.ncenters()
    }

    /// Get pq_compressed_data
    pub fn pq_compressed_data(&self) -> &PQCompressedData {
        &self.pq_compressed_data
    }

    // Get compressed vector with the given vector id from the pq_compressed_data.
    pub fn get_compressed_vector(&self, vector_id: usize) -> ANNResult<&[u8]> {
        self.pq_compressed_data.get_compressed_vector(vector_id)
    }
}

#[cfg(test)]
mod tests {

    use rstest::rstest;

    use super::*;

    fn create_pq_data(use_opq: bool) -> ANNResult<PQData> {
        let dim = 2;
        let opq_rotation_matrix = if use_opq {
            let mut opq_rotation_matrix = Vec::with_capacity(dim * dim);
            for item in 0..dim * dim {
                opq_rotation_matrix.push(item as f32 / 10.0);
            }
            opq_rotation_matrix.into_boxed_slice().into()
        } else {
            None
        };

        let pq_pivot_table = FixedChunkPQTable::new(
            dim,
            Box::new([0.0, 0.0, 1.0, 1.0]),
            Box::new([0.0, 0.0]),
            Box::new([0, 2]),
            opq_rotation_matrix,
        )
        .unwrap();
        let mut pq_compressed_data = PQCompressedData::new(3, 1).unwrap();

        let compressed_vector = [123, 111, 255];
        pq_compressed_data
            .into_dto()
            .data
            .copy_from_slice(&compressed_vector);

        PQData::new(pq_pivot_table, pq_compressed_data)
    }

    #[test]
    fn test_get_compressed_vector() {
        let dataset = create_pq_data(true).unwrap();

        let vector_id = 0;
        let result = dataset.get_compressed_vector(vector_id).unwrap();
        assert_eq!(result, &[123]);

        let vector_id = 1;
        let result = dataset.get_compressed_vector(vector_id).unwrap();
        assert_eq!(result, &[111]);

        let vector_id = 2;
        let result = dataset.get_compressed_vector(vector_id).unwrap();
        assert_eq!(result, &[255]);
    }

    #[rstest]
    fn test_get_num_chunks(#[values(true, false)] use_opq: bool) {
        let dataset = create_pq_data(use_opq).unwrap();
        assert_eq!(dataset.get_num_chunks(), 1);
    }

    #[rstest]
    fn test_get_num_centers(#[values(true, false)] use_opq: bool) {
        let dataset = create_pq_data(use_opq).unwrap();
        assert_eq!(dataset.get_num_centers(), 2);
    }
}
