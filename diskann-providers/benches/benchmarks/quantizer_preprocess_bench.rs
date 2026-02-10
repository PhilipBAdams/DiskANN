/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmark comparing the old `FixedChunkPQTable::preprocess_query +
//! populate_chunk_distances` path against the unified `quantizer_preprocess`
//! path that always uses `TransposedTable::process_into`.

use criterion::Criterion;
use diskann_providers::model::{
    NUM_PQ_CENTROIDS, PQScratch,
    pq::{FixedChunkPQTable, PQCompressedData, PQData},
};
use diskann_vector::distance::Metric;
use rand::Rng;

pub fn benchmark_quantizer_preprocess(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantizer-preprocess");
    group.sample_size(500);

    for &(dim, num_chunks, use_opq, label) in &[
        (128, 4, false, "128d/4ch"),
        (128, 4, true, "128d/4ch+opq"),
        (128, 32, false, "128d/32ch"),
        (768, 32, false, "768d/32ch"),
    ] {
        let mut rng = diskann_providers::utils::create_rnd_from_seed(42);

        let pq_table: Vec<f32> = (0..NUM_PQ_CENTROIDS * dim).map(|_| rng.random()).collect();
        let centroids: Vec<f32> = (0..dim).map(|_| rng.random()).collect();
        let chunk_size = dim / num_chunks;
        let chunk_offsets: Vec<usize> = (0..=num_chunks).map(|i| i * chunk_size).collect();
        let opq = if use_opq {
            let mut mat = vec![0.0f32; dim * dim];
            for i in 0..dim {
                mat[i * dim + i] = 1.0;
            }
            Some(mat.into_boxed_slice())
        } else {
            None
        };

        let table = FixedChunkPQTable::new(
            dim,
            pq_table.clone().into(),
            centroids.clone().into(),
            chunk_offsets.clone().into(),
            opq.clone(),
        )
        .unwrap();

        let query: Vec<f32> = (0..dim).map(|_| rng.random()).collect();

        // Old path: preprocess_query + populate_chunk_distances on FixedChunkPQTable
        group.bench_function(format!("Fixed {label}"), |b| {
            let mut rotated = query.clone();
            let mut scratch = vec![0.0f32; num_chunks * NUM_PQ_CENTROIDS];
            b.iter(|| {
                rotated.copy_from_slice(&query);
                table.preprocess_query(&mut rotated);
                table
                    .populate_chunk_distances(&rotated, &mut scratch)
                    .unwrap();
            });
        });

        // New path: PQData (always transposed) + quantizer_preprocess
        let compressed = PQCompressedData::new(1, num_chunks).unwrap();
        let pq_data = PQData::new(
            FixedChunkPQTable::new(
                dim,
                pq_table.into(),
                centroids.into(),
                chunk_offsets.into(),
                opq,
            )
            .unwrap(),
            compressed,
        )
        .unwrap();
        let mut pq_scratch = PQScratch::new(512, dim, num_chunks, NUM_PQ_CENTROIDS).unwrap();

        group.bench_function(format!("Transposed {label}"), |b| {
            b.iter(|| {
                pq_scratch.set::<f32>(dim, &query, 1.0).unwrap();
                diskann_providers::model::pq::quantizer_preprocess(
                    &mut pq_scratch,
                    &pq_data,
                    Metric::L2,
                    &[],
                )
                .unwrap();
            });
        });
    }

    group.finish();
}
