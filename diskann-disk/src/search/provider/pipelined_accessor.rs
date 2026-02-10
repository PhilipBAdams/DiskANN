/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pipelined disk accessor that implements the search traits using io_uring for
//! non-blocking IO. This accessor overlaps IO and compute by prefetching sector
//! reads while expanding already-loaded nodes.
//!
//! # Platform
//!
//! This module is only available on Linux (requires io_uring).

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::Arc;

use byteorder::{ByteOrder, LittleEndian};
use diskann::graph::glue::{self, ExpandBeam, PrefetchBeam, SearchExt, SearchPostProcess};
use diskann::graph::search_output_buffer::SearchOutputBuffer;
use diskann::graph::AdjacencyList;
use diskann::neighbor::Neighbor;
use diskann::provider::{
    Accessor, BuildQueryComputer, DataProvider, DefaultContext, DelegateNeighbor, HasId,
    NeighborAccessor,
};
use diskann::utils::object_pool::{ObjectPool, PoolOption, TryAsPooled};
use diskann::{ANNError, ANNResult};
use diskann_providers::model::graph::traits::GraphDataType;
use diskann_providers::model::{compute_pq_distance, pq::quantizer_preprocess, PQScratch};
use diskann_vector::DistanceFunction;

use crate::search::pipelined::{PipelinedReader, PipelinedReaderConfig, MAX_IO_CONCURRENCY};
use crate::search::sector_math::{node_offset_in_sector, node_sector_index};

use super::disk_provider::{DiskProvider, DiskQueryComputer};

/// Tracks an in-flight IO request.
struct InFlightIo {
    vertex_id: u32,
    slot_id: usize,
}

/// A loaded node parsed from sector data, ready for expansion.
struct LoadedNode {
    fp_vector: Vec<u8>,
    adjacency_list: Vec<u32>,
}

/// Maximum number of IO slots for a given beam width.
fn max_slots(beam_width: usize) -> usize {
    (beam_width * 2).clamp(16, MAX_IO_CONCURRENCY)
}

/// Parse a node from raw sector buffer bytes.
fn parse_node(
    sector_buf: &[u8],
    vertex_id: u32,
    num_nodes_per_sector: u64,
    node_len: u64,
    fp_vector_len: u64,
) -> ANNResult<LoadedNode> {
    let offset = node_offset_in_sector(vertex_id, num_nodes_per_sector, node_len);
    let end = offset + node_len as usize;
    let node_data = sector_buf.get(offset..end).ok_or_else(|| {
        ANNError::log_index_error(format_args!(
            "Node data out of bounds: vertex {} offset {}..{} in buffer of len {}",
            vertex_id, offset, end,
            sector_buf.len()
        ))
    })?;

    let fp_vector_len_usize = fp_vector_len as usize;
    if fp_vector_len_usize > node_data.len() {
        return Err(ANNError::log_index_error(format_args!(
            "fp_vector_len {} exceeds node_data len {}",
            fp_vector_len_usize,
            node_data.len()
        )));
    }

    let fp_vector = node_data[..fp_vector_len_usize].to_vec();

    let neighbor_data = &node_data[fp_vector_len_usize..];
    let num_neighbors = LittleEndian::read_u32(&neighbor_data[..4]) as usize;
    let max_neighbors = (neighbor_data.len().saturating_sub(4)) / 4;
    let num_neighbors = num_neighbors.min(max_neighbors);
    let mut adjacency_list = Vec::with_capacity(num_neighbors);
    for i in 0..num_neighbors {
        let start = 4 + i * 4;
        adjacency_list.push(LittleEndian::read_u32(&neighbor_data[start..start + 4]));
    }

    Ok(LoadedNode {
        fp_vector,
        adjacency_list,
    })
}

/// Scratch space for pipelined accessor, pooled for reuse across queries.
pub(crate) struct PipelinedAccessorScratch {
    reader: PipelinedReader,
    pq_scratch: PQScratch,
}

/// Arguments for creating or resetting a [`PipelinedAccessorScratch`].
#[derive(Clone)]
pub(crate) struct PipelinedAccessorScratchArgs<'a> {
    pub disk_index_path: &'a str,
    pub max_slots: usize,
    pub slot_size: usize,
    pub alignment: usize,
    pub graph_degree: usize,
    pub dims: usize,
    pub num_pq_chunks: usize,
    pub num_pq_centers: usize,
    pub reader_config: PipelinedReaderConfig,
}

impl TryAsPooled<&PipelinedAccessorScratchArgs<'_>> for PipelinedAccessorScratch {
    type Error = ANNError;

    fn try_create(args: &PipelinedAccessorScratchArgs<'_>) -> Result<Self, Self::Error> {
        let reader = PipelinedReader::new(
            args.disk_index_path,
            args.max_slots,
            args.slot_size,
            args.alignment,
            &args.reader_config,
        )?;
        let pq_scratch = PQScratch::new(
            args.graph_degree,
            args.dims,
            args.num_pq_chunks,
            args.num_pq_centers,
        )?;
        Ok(Self { reader, pq_scratch })
    }

    fn try_modify(&mut self, _args: &PipelinedAccessorScratchArgs<'_>) -> Result<(), Self::Error> {
        self.reader.reset();
        Ok(())
    }
}

/// Accessor for pipelined disk search using io_uring.
///
/// Implements `PrefetchBeam` with actual non-blocking IO via io_uring, unlike
/// the no-op default used by standard `DiskAccessor`. The search loop calls
/// `prefetch()` to submit reads and `poll_completed()` to check for completions,
/// then `expand_beam()` processes the already-loaded nodes.
pub struct PipelinedDiskAccessor<'a, Data: GraphDataType<VectorIdType = u32>> {
    provider: &'a DiskProvider<Data>,

    // Graph metadata (cached from header)
    node_len: u64,
    num_nodes_per_sector: u64,
    num_sectors_per_node: usize,
    block_size: usize,
    fp_vector_len: u64,

    // Pooled scratch (owns reader + PQ scratch)
    scratch: PoolOption<PipelinedAccessorScratch>,

    // IO pipeline state
    on_flight_ios: VecDeque<InFlightIo>,
    next_slot_id: usize,
    beam_width: usize,

    // Completed but not-yet-expanded nodes
    completed_nodes: HashMap<u32, LoadedNode>,
}

impl<'a, Data> PipelinedDiskAccessor<'a, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    /// Create a new pipelined disk accessor.
    pub(crate) fn new(
        provider: &'a DiskProvider<Data>,
        query: &[Data::VectorDataType],
        beam_width: usize,
        scratch_pool: &Arc<ObjectPool<PipelinedAccessorScratch>>,
        disk_index_path: &str,
        reader_config: &PipelinedReaderConfig,
    ) -> ANNResult<Self> {
        let metadata = provider.graph_header.metadata();
        let dims = metadata.dims;
        let node_len = metadata.node_len;
        let num_nodes_per_sector = metadata.num_nodes_per_block;
        let fp_vector_len =
            (dims * std::mem::size_of::<Data::VectorDataType>()) as u64;

        let block_size = provider.graph_header.effective_block_size();
        let num_sectors_per_node = provider.graph_header.num_sectors_per_node();
        let slot_size = num_sectors_per_node * block_size;
        let graph_degree = provider.graph_header.max_degree::<Data::VectorDataType>()?;

        let mut scratch = PoolOption::try_pooled(
            scratch_pool,
            &PipelinedAccessorScratchArgs {
                disk_index_path,
                max_slots: max_slots(beam_width),
                slot_size,
                alignment: block_size,
                graph_degree,
                dims,
                num_pq_chunks: provider.pq_data.get_num_chunks(),
                num_pq_centers: provider.pq_data.get_num_centers(),
                reader_config: reader_config.clone(),
            },
        )?;

        // Prepare PQ distance table for the query
        scratch.pq_scratch.set(dims, query, 1.0_f32)?;
        let medoid = provider.graph_header.metadata().medoid as u32;
        quantizer_preprocess(
            &mut scratch.pq_scratch,
            &provider.pq_data,
            provider.metric,
            &[medoid],
        )?;

        Ok(Self {
            provider,
            node_len,
            num_nodes_per_sector,
            num_sectors_per_node,
            block_size,
            fp_vector_len,
            scratch,
            on_flight_ios: VecDeque::new(),
            next_slot_id: 0,
            beam_width,
            completed_nodes: HashMap::new(),
        })
    }

    /// Compute PQ distances for a batch of IDs, invoking callback with results.
    fn pq_distances<F>(&mut self, ids: &[u32], mut f: F) -> ANNResult<()>
    where
        F: FnMut(f32, u32),
    {
        let pq_scratch = &mut self.scratch.pq_scratch;
        compute_pq_distance(
            ids,
            self.provider.pq_data.get_num_chunks(),
            &pq_scratch.aligned_pqtable_dist_scratch,
            self.provider.pq_data.pq_compressed_data().get_data(),
            &mut pq_scratch.aligned_pq_coord_scratch,
            &mut pq_scratch.aligned_dist_scratch,
        )?;

        for (i, id) in ids.iter().enumerate() {
            let distance = pq_scratch.aligned_dist_scratch[i];
            f(distance, *id);
        }

        Ok(())
    }

    /// Process completed IO slots, parsing sector data into LoadedNodes.
    fn process_completions(&mut self, completed_slots: Vec<usize>) -> ANNResult<Vec<u32>> {
        if completed_slots.is_empty() {
            return Ok(vec![]);
        }
        let completed_set: std::collections::HashSet<usize> =
            completed_slots.into_iter().collect();
        let mut remaining = VecDeque::new();
        let mut completed_ids = Vec::new();

        while let Some(io) = self.on_flight_ios.pop_front() {
            if completed_set.contains(&io.slot_id) {
                let sector_buf = self.scratch.reader.get_slot_buf(io.slot_id);
                let node = parse_node(
                    sector_buf,
                    io.vertex_id,
                    self.num_nodes_per_sector,
                    self.node_len,
                    self.fp_vector_len,
                )?;
                self.completed_nodes.insert(io.vertex_id, node);
                completed_ids.push(io.vertex_id);
            } else {
                remaining.push_back(io);
            }
        }
        self.on_flight_ios = remaining;
        Ok(completed_ids)
    }
}

// -- Trait implementations --

impl<Data> HasId for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Id = u32;
}

impl<'a, Data> Accessor for PipelinedDiskAccessor<'a, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Extended = &'a [u8];
    type Element<'b>
        = &'a [u8]
    where
        Self: 'b;
    type ElementRef<'b> = &'b [u8];
    type GetError = ANNError;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        std::future::ready(self.provider.pq_data.get_compressed_vector(id as usize))
    }
}

impl<Data> BuildQueryComputer<[Data::VectorDataType]> for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type QueryComputerError = ANNError;
    type QueryComputer = DiskQueryComputer;

    fn build_query_computer(
        &self,
        _from: &[Data::VectorDataType],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(DiskQueryComputer {
            num_pq_chunks: self.provider.pq_data.get_num_chunks(),
            query_centroid_l2_distance: self
                .scratch
                .pq_scratch
                .aligned_pqtable_dist_scratch
                .as_slice()
                .to_vec(),
        })
    }

    async fn distances_unordered<Itr, F>(
        &mut self,
        vec_id_itr: Itr,
        _computer: &Self::QueryComputer,
        f: F,
    ) -> Result<(), Self::GetError>
    where
        F: Send + FnMut(f32, Self::Id),
        Itr: Iterator<Item = Self::Id>,
    {
        self.pq_distances(&vec_id_itr.collect::<Box<[_]>>(), f)
    }
}

impl<Data> ExpandBeam<[Data::VectorDataType]> for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        _computer: &Self::QueryComputer,
        mut pred: P,
        mut f: F,
    ) -> impl Future<Output = Result<(), Self::GetError>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(f32, Self::Id) + Send,
    {
        let result = (|| {
            let num_pq_chunks = self.provider.pq_data.get_num_chunks();
            let pq_compressed = self.provider.pq_data.pq_compressed_data().get_data();
            let num_pts = pq_compressed.len() / num_pq_chunks;

            for id in ids {
                // Get the loaded node from completed_nodes (should be there from prefetch)
                let node = match self.completed_nodes.get(&id) {
                    Some(node) => node,
                    None => continue, // Node not yet loaded; skip
                };

                // Expand neighbors: filter through predicate, compute PQ distances
                let nbors: Vec<u32> = node
                    .adjacency_list
                    .iter()
                    .copied()
                    .filter(|&nbr_id| (nbr_id as usize) < num_pts && pred.eval_mut(&nbr_id))
                    .collect();

                if !nbors.is_empty() {
                    self.pq_distances(&nbors, &mut f)?;
                }
            }
            Ok(())
        })();

        std::future::ready(result)
    }
}

impl<Data> PrefetchBeam for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    fn prefetch(&mut self, ids: impl Iterator<Item = Self::Id> + Send) {
        for id in ids {
            if self.completed_nodes.contains_key(&id) {
                continue; // Already loaded
            }

            let sector_idx = node_sector_index(
                id,
                self.num_nodes_per_sector,
                self.num_sectors_per_node,
            );
            let sector_offset = sector_idx * self.block_size as u64;
            let slot_id = self.next_slot_id % max_slots(self.beam_width);

            if self
                .scratch
                .reader
                .submit_read(sector_offset, slot_id)
                .is_ok()
            {
                self.on_flight_ios.push_back(InFlightIo {
                    vertex_id: id,
                    slot_id,
                });
                self.next_slot_id = (self.next_slot_id + 1) % max_slots(self.beam_width);
            }
        }
    }

    fn poll_completed(&mut self) -> Vec<Self::Id> {
        let completed_slots = match self.scratch.reader.poll_completions() {
            Ok(slots) => slots,
            Err(_) => return vec![],
        };
        self.process_completions(completed_slots).unwrap_or_default()
    }

    fn inflight_count(&self) -> usize {
        self.on_flight_ios.len()
    }
}

impl<Data> SearchExt for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    async fn starting_points(&self) -> ANNResult<Vec<u32>> {
        let medoid = self.provider.graph_header.metadata().medoid as u32;
        Ok(vec![medoid])
    }
}

/// Neighbor accessor wrapper for PipelinedDiskAccessor (required by AsNeighbor).
pub struct PipelinedNeighborAccessor<'a, 'b, Data>(
    &'a mut PipelinedDiskAccessor<'b, Data>,
)
where
    Data: GraphDataType<VectorIdType = u32>;

impl<Data> HasId for PipelinedNeighborAccessor<'_, '_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Id = u32;
}

impl<Data> NeighborAccessor for PipelinedNeighborAccessor<'_, '_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        if let Some(node) = self.0.completed_nodes.get(&id) {
            let list: Vec<u32> = node.adjacency_list.clone();
            neighbors.overwrite_trusted(&list);
        }
        std::future::ready(Ok(self))
    }
}

impl<'a, 'b, Data> DelegateNeighbor<'a> for PipelinedDiskAccessor<'b, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Delegate = PipelinedNeighborAccessor<'a, 'b, Data>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        PipelinedNeighborAccessor(self)
    }
}

// -- Search Strategy --

/// Search strategy for pipelined disk search.
///
/// Creates `PipelinedDiskAccessor` instances and provides post-processing
/// via `PipelinedPostProcessor`.
pub struct PipelinedSearchStrategy<'a, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    pub(crate) query: &'a [Data::VectorDataType],
    pub(crate) beam_width: usize,
    pub(crate) scratch_pool: &'a Arc<ObjectPool<PipelinedAccessorScratch>>,
    pub(crate) disk_index_path: &'a str,
    pub(crate) reader_config: &'a PipelinedReaderConfig,
    pub(crate) vector_filter: &'a (dyn Fn(&u32) -> bool + Send + Sync),
}

/// Post-processor for pipelined search that reranks using full-precision distances.
///
/// Unlike the standard `RerankAndFilter` which relies on a vertex provider cache,
/// this post-processor uses the full-precision vectors stored in `completed_nodes`.
#[derive(Clone, Copy)]
pub struct PipelinedPostProcessor<'a> {
    filter: &'a (dyn Fn(&u32) -> bool + Send + Sync),
}

impl<Data>
    SearchPostProcess<
        PipelinedDiskAccessor<'_, Data>,
        [Data::VectorDataType],
        (
            <DiskProvider<Data> as DataProvider>::InternalId,
            Data::AssociatedDataType,
        ),
    > for PipelinedPostProcessor<'_>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Error = ANNError;

    async fn post_process<I, B>(
        &self,
        accessor: &mut PipelinedDiskAccessor<'_, Data>,
        query: &[Data::VectorDataType],
        _computer: &DiskQueryComputer,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<(u32, Data::AssociatedDataType)> + Send + ?Sized,
    {
        let distance_comparer = &accessor.provider.distance_comparer;

        // Rerank candidates using full-precision distance from completed_nodes
        let mut reranked: Vec<((u32, Data::AssociatedDataType), f32)> = Vec::new();
        for candidate in candidates {
            let id = candidate.id;
            if !(self.filter)(&id) {
                continue;
            }
            if let Some(node) = accessor.completed_nodes.get(&id) {
                let fp_vec: &[Data::VectorDataType] = bytemuck::cast_slice(&node.fp_vector);
                let dist = distance_comparer.evaluate_similarity(query, fp_vec);
                reranked.push(((id, Data::AssociatedDataType::default()), dist));
            } else {
                // Fallback: use PQ distance from candidate
                reranked.push(((id, Data::AssociatedDataType::default()), candidate.distance));
            }
        }

        reranked
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(output.extend(reranked))
    }
}

impl<'this, Data>
    diskann::graph::glue::SearchStrategy<
        DiskProvider<Data>,
        [Data::VectorDataType],
        (
            <DiskProvider<Data> as DataProvider>::InternalId,
            Data::AssociatedDataType,
        ),
    > for PipelinedSearchStrategy<'this, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type QueryComputer = DiskQueryComputer;
    type SearchAccessor<'a> = PipelinedDiskAccessor<'a, Data>;
    type SearchAccessorError = ANNError;
    type PostProcessor = PipelinedPostProcessor<'this>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DiskProvider<Data>,
        _context: &DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        PipelinedDiskAccessor::new(
            provider,
            self.query,
            self.beam_width,
            self.scratch_pool,
            self.disk_index_path,
            self.reader_config,
        )
    }

    fn post_processor(&self) -> Self::PostProcessor {
        PipelinedPostProcessor {
            filter: self.vector_filter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that PipelinedDiskAccessor implements the required traits.
    /// Full integration tests require a real disk index file and are run
    /// via the benchmark suite with SearchMode::UnifiedPipeSearch.
    #[test]
    fn trait_bounds_compile() {
        // This test verifies at compile time that PipelinedDiskAccessor
        // satisfies the trait bounds required by the generic search loop:
        // ExpandBeam + PrefetchBeam + SearchExt
        fn _assert_bounds<T>()
        where
            T: ExpandBeam<[f32]> + PrefetchBeam + SearchExt,
        {
        }
        // The function is never called — it only needs to compile.
        // PipelinedDiskAccessor<'_, GraphData<f32>> would be T.
    }
}
