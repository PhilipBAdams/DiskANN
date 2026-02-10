/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_linalg::Transpose;
use diskann_vector::distance::Metric;

use super::fixed_chunk_pq_table::compute_pq_distance;
use crate::{
    model::{PQData, PQScratch},
    utils::BridgeErr,
};

/// Preprocesses the query vector for PQ distance calculations.
/// This function rotates the query vector and prepares the PQ table distances
/// for efficient computation during search operations.
pub fn quantizer_preprocess(
    pq_scratch: &mut PQScratch,
    pq_data: &PQData,
    metric: Metric,
    id_to_calculate_pq_distance: &[u32],
) -> ANNResult<()> {
    let pq_table = pq_data.pq_table();
    let dim = pq_table.dim();

    // Step 1: Subtract dataset centroids from the query vector.
    let query = &mut pq_scratch.rotated_query[..dim];
    for (q, &c) in query.iter_mut().zip(pq_table.centroids().iter()) {
        *q -= c;
    }

    // Step 2: Apply OPQ rotation if present.
    if let Some(rotation_matrix) = pq_table.opq_rotation_matrix() {
        let mut temp_result = vec![0.0f32; dim];
        diskann_linalg::sgemm(
            Transpose::None,
            Transpose::None,
            1,
            dim,
            dim,
            1.0,
            &pq_scratch.rotated_query[..dim],
            rotation_matrix,
            None,
            &mut temp_result,
        );
        pq_scratch.rotated_query[..dim].copy_from_slice(&temp_result);
    }

    // Step 3: Build the distance lookup table using the fast transposed path.
    let table = pq_table.transposed_table();
    let expected_len = table.ncenters() * table.nchunks();
    let dst = diskann_utils::views::MutMatrixView::try_from(
        &mut pq_scratch.aligned_pqtable_dist_scratch.as_mut_slice()[..expected_len],
        table.nchunks(),
        table.ncenters(),
    )
    .bridge_err()?;

    match metric {
        Metric::L2 | Metric::Cosine | Metric::CosineNormalized => {
            table.process_into::<diskann_quantization::distances::SquaredL2>(
                &pq_scratch.rotated_query[..dim],
                dst,
            );
        }
        Metric::InnerProduct => {
            table.process_into::<diskann_quantization::distances::InnerProduct>(
                &pq_scratch.rotated_query[..dim],
                dst,
            );
        }
    }

    // Step 4: Compute PQ distances for the given IDs.
    compute_pq_distance(
        id_to_calculate_pq_distance,
        pq_data.get_num_chunks(),
        &pq_scratch.aligned_pqtable_dist_scratch,
        pq_data.pq_compressed_data().get_data(),
        &mut pq_scratch.aligned_pq_coord_scratch,
        &mut pq_scratch.aligned_dist_scratch,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use diskann_vector::distance::Metric;
    use rand::Rng;

    use super::*;
    use crate::model::{
        PQScratch,
        pq::{FixedChunkPQTable, PQCompressedData},
    };

    /// Build a PQData + PQScratch + query for a given config.
    /// Returns `(pq_data, pq_scratch, query)`.
    fn setup(
        dim: usize,
        num_chunks: usize,
        use_opq: bool,
        non_zero_centroids: bool,
    ) -> (PQData, PQScratch, Vec<f32>) {
        let mut rng = crate::utils::create_rnd_in_tests();
        let num_centers = 256;

        let pq_table: Vec<f32> = (0..num_centers * dim).map(|_| rng.random()).collect();
        let centroids: Vec<f32> = if non_zero_centroids {
            (0..dim).map(|_| rng.random()).collect()
        } else {
            vec![0.0; dim]
        };
        let chunk_size = dim / num_chunks;
        let chunk_offsets: Vec<usize> = (0..=num_chunks).map(|i| i * chunk_size).collect();
        let opq = if use_opq {
            // Build a simple rotation matrix (identity + small perturbation).
            let mut mat = vec![0.0f32; dim * dim];
            for i in 0..dim {
                mat[i * dim + i] = 1.0;
                if i + 1 < dim {
                    mat[i * dim + i + 1] = 0.01;
                }
            }
            Some(mat.into_boxed_slice())
        } else {
            None
        };

        let table = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            opq,
        )
        .unwrap();

        // Build compressed data for 2 vectors.
        let num_vectors = 2;
        let mut compressed = PQCompressedData::new(num_vectors, num_chunks).unwrap();
        let data = compressed.into_dto().data;
        for b in data.iter_mut() {
            *b = rng.random();
        }

        let pq_data = PQData::new(table, compressed).unwrap();
        let pq_scratch =
            PQScratch::new(512, dim, num_chunks, num_centers).unwrap();
        let query: Vec<f32> = (0..dim).map(|_| rng.random()).collect();
        (pq_data, pq_scratch, query)
    }

    /// Compute the distance table using the old `FixedChunkPQTable` code path
    /// (centroid subtraction + OPQ rotation + populate_chunk_distances_impl).
    fn reference_distances(
        dim: usize,
        num_chunks: usize,
        use_opq: bool,
        non_zero_centroids: bool,
        metric: Metric,
        query: &[f32],
    ) -> Vec<f32> {
        let mut rng = crate::utils::create_rnd_in_tests();
        let num_centers = 256;

        let pq_table: Vec<f32> = (0..num_centers * dim).map(|_| rng.random()).collect();
        let centroids: Vec<f32> = if non_zero_centroids {
            (0..dim).map(|_| rng.random()).collect()
        } else {
            vec![0.0; dim]
        };
        let chunk_size = dim / num_chunks;
        let chunk_offsets: Vec<usize> = (0..=num_chunks).map(|i| i * chunk_size).collect();
        let opq = if use_opq {
            let mut mat = vec![0.0f32; dim * dim];
            for i in 0..dim {
                mat[i * dim + i] = 1.0;
                if i + 1 < dim {
                    mat[i * dim + i + 1] = 0.01;
                }
            }
            Some(mat.into_boxed_slice())
        } else {
            None
        };

        let table = FixedChunkPQTable::new(
            dim,
            pq_table.into(),
            centroids.into(),
            chunk_offsets.into(),
            opq,
        )
        .unwrap();

        let mut rotated_query = query.to_vec();
        table.preprocess_query(&mut rotated_query);

        let mut dist_scratch = vec![0.0f32; num_chunks * num_centers];
        match metric {
            Metric::L2 | Metric::Cosine | Metric::CosineNormalized => {
                table
                    .populate_chunk_distances(&rotated_query, &mut dist_scratch)
                    .unwrap();
            }
            Metric::InnerProduct => {
                table
                    .populate_chunk_inner_products(&rotated_query, &mut dist_scratch)
                    .unwrap();
            }
        }
        dist_scratch
    }

    /// Compute the distance table through the new unified `quantizer_preprocess` path.
    fn new_path_distances(
        pq_data: &PQData,
        pq_scratch: &mut PQScratch,
        query: &[f32],
        metric: Metric,
    ) -> Vec<f32> {
        let dim = pq_data.pq_table().dim();
        pq_scratch.set::<f32>(dim, query, 1.0).unwrap();
        quantizer_preprocess(pq_scratch, pq_data, metric, &[]).unwrap();
        let n = pq_data.get_num_chunks() * pq_data.get_num_centers();
        pq_scratch.aligned_pqtable_dist_scratch.as_slice()[..n].to_vec()
    }

    #[test]
    fn transposed_matches_fixed_no_opq_zero_centroids_l2() {
        let (pq_data, mut scratch, query) = setup(128, 4, false, false);
        let reference = reference_distances(128, 4, false, false, Metric::L2, &query);
        let result = new_path_distances(&pq_data, &mut scratch, &query, Metric::L2);
        for (r, n) in reference.iter().zip(result.iter()) {
            assert_relative_eq!(r, n, epsilon = 1e-4, max_relative = 1e-4);
        }
    }

    #[test]
    fn transposed_matches_fixed_no_opq_nonzero_centroids_l2() {
        let (pq_data, mut scratch, query) = setup(128, 4, false, true);
        let reference = reference_distances(128, 4, false, true, Metric::L2, &query);
        let result = new_path_distances(&pq_data, &mut scratch, &query, Metric::L2);
        for (r, n) in reference.iter().zip(result.iter()) {
            assert_relative_eq!(r, n, epsilon = 1e-4, max_relative = 1e-4);
        }
    }

    #[test]
    fn transposed_matches_fixed_with_opq_l2() {
        let (pq_data, mut scratch, query) = setup(128, 4, true, true);
        let reference = reference_distances(128, 4, true, true, Metric::L2, &query);
        let result = new_path_distances(&pq_data, &mut scratch, &query, Metric::L2);
        for (r, n) in reference.iter().zip(result.iter()) {
            assert_relative_eq!(r, n, epsilon = 1e-4, max_relative = 1e-4);
        }
    }

    #[test]
    fn transposed_matches_fixed_with_opq_ip() {
        let (pq_data, mut scratch, query) = setup(128, 4, true, true);
        let reference = reference_distances(128, 4, true, true, Metric::InnerProduct, &query);
        let result = new_path_distances(&pq_data, &mut scratch, &query, Metric::InnerProduct);
        for (r, n) in reference.iter().zip(result.iter()) {
            assert_relative_eq!(r, n, epsilon = 1e-4, max_relative = 1e-4);
        }
    }
}
