/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Microbenchmark comparing centroid-major vs dimension-major
//! `populate_chunk_distances` loop orders.

use criterion::Criterion;
use diskann_providers::model::{NUM_PQ_CENTROIDS, pq::FixedChunkPQTable};
use rand::Rng;

/// Build a random FixedChunkPQTable with the given parameters.
fn make_table(dim: usize, num_chunks: usize) -> (FixedChunkPQTable, Vec<f32>, Vec<f32>) {
    let mut rng = diskann_providers::utils::create_rnd_from_seed(42);

    let pq_table: Vec<f32> = (0..NUM_PQ_CENTROIDS * dim).map(|_| rng.random()).collect();
    let centroids: Vec<f32> = (0..dim).map(|_| rng.random()).collect();
    let chunk_size = dim / num_chunks;
    let chunk_offsets: Vec<usize> = (0..=num_chunks).map(|i| i * chunk_size).collect();

    let table = FixedChunkPQTable::new(
        dim,
        pq_table.into(),
        centroids.into(),
        chunk_offsets.into(),
        None,
    )
    .unwrap();

    let tables_t = table.build_transposed_table();
    let query: Vec<f32> = (0..dim).map(|_| rng.random()).collect();
    (table, tables_t, query)
}

/// Bench all variants for a given (dim, num_chunks) configuration.
fn bench_config(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, dim: usize, num_chunks: usize) {
    let (table, tables_t, query) = make_table(dim, num_chunks);
    let scratch_len = num_chunks * NUM_PQ_CENTROIDS;
    let label = format!("({dim}d, {num_chunks} chunks)");

    group.bench_function(format!("centroid-major L2 {label}"), |b| {
        let mut scratch = vec![0.0f32; scratch_len];
        b.iter(|| table.populate_chunk_distances(&query, &mut scratch).unwrap());
    });

    group.bench_function(format!("dim-major L2 {label}"), |b| {
        let mut scratch = vec![0.0f32; scratch_len];
        b.iter(|| table.populate_chunk_distances_dim_major_l2(&query, &mut scratch).unwrap());
    });

    group.bench_function(format!("transposed L2 {label}"), |b| {
        let mut scratch = vec![0.0f32; scratch_len];
        b.iter(|| table.populate_chunk_distances_transposed_l2(&tables_t, &query, &mut scratch).unwrap());
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function(format!("transposed-avx2 L2 {label}"), |b| {
        let mut scratch = vec![0.0f32; scratch_len];
        b.iter(|| table.populate_chunk_distances_transposed_simd_l2(&tables_t, &query, &mut scratch).unwrap());
    });

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use diskann_providers::common::AlignedBoxWithSlice;

        let tables_t_aligned = table.build_transposed_table_aligned();
        let mut scratch_aligned = AlignedBoxWithSlice::<f32>::new(scratch_len, 64).unwrap();

        group.bench_function(format!("transposed-avx512-nt L2 {label}"), |b| {
            b.iter(|| {
                table
                    .populate_chunk_distances_transposed_avx512_l2(
                        tables_t_aligned.as_slice(),
                        &query,
                        scratch_aligned.as_mut_slice(),
                    )
                    .unwrap()
            });
        });

        group.bench_function(format!("transposed-avx512 L2 {label}"), |b| {
            let mut scratch = vec![0.0f32; scratch_len];
            b.iter(|| {
                table
                    .populate_chunk_distances_transposed_avx512_regular_l2(
                        &tables_t,
                        &query,
                        &mut scratch,
                    )
                    .unwrap()
            });
        });
    }
}

pub fn benchmark_populate_chunk_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("populate-chunk-distances");
    group.sample_size(500);

    bench_config(&mut group, 128, 4);   // typical SIFT
    bench_config(&mut group, 128, 32);  // many-chunk
    bench_config(&mut group, 768, 32);  // high-dimensional text embeddings

    group.finish();
}
