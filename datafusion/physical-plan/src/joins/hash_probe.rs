// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`HashProbeExec`] Partitioned Hash Join Operator

use std::fmt;
use std::sync::Arc;
use std::task::Poll;
use std::{any::Any, usize, vec};

use super::{
    utils::{OnceAsync, OnceFut},
    PartitionMode,
};
use crate::ExecutionPlanProperties;
use crate::{
    coalesce_partitions::CoalescePartitionsExec,
    common::can_project,
    execution_mode_from_children, handle_state,
    hash_utils::create_hashes,
    joins::utils::{
        adjust_indices_by_join_type, adjust_right_output_partitioning,
        apply_join_filter_to_indices, build_batch_from_indices, build_join_schema,
        check_join_is_valid, estimate_join_statistics, get_final_indices_from_bit_map,
        need_produce_result_in_final, partitioned_join_output_partitioning,
        BuildProbeJoinMetrics, ColumnIndex, JoinFilter, JoinHashMap, JoinHashMapOffset,
        JoinHashMapType, JoinOn, JoinOnRef, StatefulStreamResult,
    },
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
    DisplayAs, DisplayFormatType, Distribution, ExecutionMode, ExecutionPlan,
    Partitioning, PlanProperties, RecordBatchStream, SendableRecordBatchStream,
    Statistics,
};

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBufferBuilder, PrimitiveArray, UInt32Array,
    UInt64Array,
};
use arrow::compute::kernels::cmp::{eq, not_distinct};
use arrow::compute::{and, concat_batches, take, FilterBuilder};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::util::bit_util;
use arrow_array::cast::downcast_array;
use arrow_schema::ArrowError;
use datafusion_common::{
    internal_datafusion_err, internal_err, plan_err, project_schema, DataFusionError,
    JoinSide, JoinType, Result,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::equivalence::{
    join_equivalence_properties, ProjectionMapping,
};
use datafusion_physical_expr::expressions::UnKnownColumn;
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprRef};

use ahash::RandomState;
use futures::{ready, Stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};

/// HashTable and input data for the left (build side) of a join
#[derive(Debug, Clone)]
pub struct JoinLeftData {
    /// The hash table with indices into `batch`
    hash_map: JoinHashMap,
    /// The input rows for the build side
    batch: RecordBatch,
    /// Memory reservation that tracks memory used by `hash_map` hash table
    /// `batch`. Cleared on drop.
    //#[allow(dead_code)]
    // reservation: MemoryReservation,
    output_partitioning: Partitioning,
}

impl JoinLeftData {
    /// Create a new `JoinLeftData` from its parts
    pub fn new(
        hash_map: JoinHashMap,
        batch: RecordBatch,
        output_partitioning: Partitioning,
    ) -> Self {
        Self {
            hash_map,
            batch,
            output_partitioning,
        }
    }

    /// Returns the number of rows in the build side
    fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    /// return a reference to the hash map
    fn hash_map(&self) -> &JoinHashMap {
        &self.hash_map
    }

    /// returns a reference to the build side batch
    fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

/// Join execution plan: Evaluates eqijoin predicates in parallel on multiple
/// partitions using a hash table and an optional filter list to apply post
/// join.
///
/// # Join Expressions
///
/// This implementation is optimized for evaluating eqijoin predicates  (
/// `<col1> = <col2>`) expressions, which are represented as a list of `Columns`
/// in [`Self::on`].
///
/// Non-equality predicates, which can not pushed down to a join inputs (e.g.
/// `<col1> != <col2>`) are known as "filter expressions" and are evaluated
/// after the equijoin predicates.
///
/// # "Build Side" vs "Probe Side"
///
/// HashJoin takes two inputs, which are referred to as the "build" and the
/// "probe". The build side is the first child, and the probe side is the second
/// child.
///
/// The two inputs are treated differently and it is VERY important that the
/// *smaller* input is placed on the build side to minimize the work of creating
/// the hash table.
///
/// ```text
///          ┌───────────┐
///          │ HashJoin  │
///          │           │
///          └───────────┘
///              │   │
///        ┌─────┘   └─────┐
///        ▼               ▼
/// ┌────────────┐  ┌─────────────┐
/// │   Input    │  │    Input    │
/// │    [0]     │  │     [1]     │
/// └────────────┘  └─────────────┘
///
///  "build side"    "probe side"
/// ```
///
/// Execution proceeds in 2 stages:
///
/// 1. the **build phase** creates a hash table from the tuples of the build side,
/// and single concatenated batch containing data from all fetched record batches.
/// Resulting hash table stores hashed join-key fields for each row as a key, and
/// indices of corresponding rows in concatenated batch.
///
/// Hash join uses LIFO data structure as a hash table, and in order to retain
/// original build-side input order while obtaining data during probe phase, hash
/// table is updated by iterating batch sequence in reverse order -- it allows to
/// keep rows with smaller indices "on the top" of hash table, and still maintain
/// correct indexing for concatenated build-side data batch.
///
/// Example of build phase for 3 record batches:
///
///
/// ```text
///
///  Original build-side data   Inserting build-side values into hashmap    Concatenated build-side batch
///                                                                         ┌───────────────────────────┐
///                             hasmap.insert(row-hash, row-idx + offset)   │                      idx  │
///            ┌───────┐                                                    │          ┌───────┐        │
///            │ Row 1 │        1) update_hash for batch 3 with offset 0    │          │ Row 6 │    0   │
///   Batch 1  │       │           - hashmap.insert(Row 7, idx 1)           │ Batch 3  │       │        │
///            │ Row 2 │           - hashmap.insert(Row 6, idx 0)           │          │ Row 7 │    1   │
///            └───────┘                                                    │          └───────┘        │
///                                                                         │                           │
///            ┌───────┐                                                    │          ┌───────┐        │
///            │ Row 3 │        2) update_hash for batch 2 with offset 2    │          │ Row 3 │    2   │
///            │       │           - hashmap.insert(Row 5, idx 4)           │          │       │        │
///   Batch 2  │ Row 4 │           - hashmap.insert(Row 4, idx 3)           │ Batch 2  │ Row 4 │    3   │
///            │       │           - hashmap.insert(Row 3, idx 2)           │          │       │        │
///            │ Row 5 │                                                    │          │ Row 5 │    4   │
///            └───────┘                                                    │          └───────┘        │
///                                                                         │                           │
///            ┌───────┐                                                    │          ┌───────┐        │
///            │ Row 6 │        3) update_hash for batch 1 with offset 5    │          │ Row 1 │    5   │
///   Batch 3  │       │           - hashmap.insert(Row 2, idx 5)           │ Batch 1  │       │        │
///            │ Row 7 │           - hashmap.insert(Row 1, idx 6)           │          │ Row 2 │    6   │
///            └───────┘                                                    │          └───────┘        │
///                                                                         │                           │
///                                                                         └───────────────────────────┘
///
/// ```
///
/// 2. the **probe phase** where the tuples of the probe side are streamed
/// through, checking for matches of the join keys in the hash table.
///
/// ```text
///                 ┌────────────────┐          ┌────────────────┐
///                 │ ┌─────────┐    │          │ ┌─────────┐    │
///                 │ │  Hash   │    │          │ │  Hash   │    │
///                 │ │  Table  │    │          │ │  Table  │    │
///                 │ │(keys are│    │          │ │(keys are│    │
///                 │ │equi join│    │          │ │equi join│    │  Stage 2: batches from
///  Stage 1: the   │ │columns) │    │          │ │columns) │    │    the probe side are
/// *entire* build  │ │         │    │          │ │         │    │  streamed through, and
///  side is read   │ └─────────┘    │          │ └─────────┘    │   checked against the
/// into the hash   │      ▲         │          │          ▲     │   contents of the hash
///     table       │       HashJoin │          │  HashJoin      │          table
///                 └──────┼─────────┘          └──────────┼─────┘
///             ─ ─ ─ ─ ─ ─                                 ─ ─ ─ ─ ─ ─ ─
///            │                                                         │
///
///            │                                                         │
///     ┌────────────┐                                            ┌────────────┐
///     │RecordBatch │                                            │RecordBatch │
///     └────────────┘                                            └────────────┘
///     ┌────────────┐                                            ┌────────────┐
///     │RecordBatch │                                            │RecordBatch │
///     └────────────┘                                            └────────────┘
///           ...                                                       ...
///     ┌────────────┐                                            ┌────────────┐
///     │RecordBatch │                                            │RecordBatch │
///     └────────────┘                                            └────────────┘
///
///        build side                                                probe side
///
/// ```
///
/// # Example "Optimal" Plans
///
/// The differences in the inputs means that for classic "Star Schema Query",
/// the optimal plan will be a **"Right Deep Tree"** . A Star Schema Query is
/// one where there is one large table and several smaller "dimension" tables,
/// joined on `Foreign Key = Primary Key` predicates.
///
/// A "Right Deep Tree" looks like this large table as the probe side on the
/// lowest join:
///
/// ```text
///             ┌───────────┐
///             │ HashJoin  │
///             │           │
///             └───────────┘
///                 │   │
///         ┌───────┘   └──────────┐
///         ▼                      ▼
/// ┌───────────────┐        ┌───────────┐
/// │ small table 1 │        │ HashJoin  │
/// │  "dimension"  │        │           │
/// └───────────────┘        └───┬───┬───┘
///                   ┌──────────┘   └───────┐
///                   │                      │
///                   ▼                      ▼
///           ┌───────────────┐        ┌───────────┐
///           │ small table 2 │        │ HashJoin  │
///           │  "dimension"  │        │           │
///           └───────────────┘        └───┬───┬───┘
///                               ┌────────┘   └────────┐
///                               │                     │
///                               ▼                     ▼
///                       ┌───────────────┐     ┌───────────────┐
///                       │ small table 3 │     │  large table  │
///                       │  "dimension"  │     │    "fact"     │
///                       └───────────────┘     └───────────────┘
/// ```
#[derive(Debug)]
pub struct HashProbeExec {
    /// right (probe) side which are filtered by the hash table
    pub right: Arc<dyn ExecutionPlan>,
    /// Set of equijoin columns from the relations: `(left_col, right_col)`
    pub on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    /// Filters which are applied while finding matching rows
    pub filter: Option<JoinFilter>,
    /// How the join is performed (`OUTER`, `INNER`, etc)
    pub join_type: JoinType,
    /// The schema after join. Please be careful when using this schema,
    /// if there is a projection, the schema isn't the same as the output schema.
    join_schema: SchemaRef,
    /// Future that consumes left input and builds the hash table
    left_fut: OnceAsync<JoinLeftData>,
    /// Shared the `RandomState` for the hashing algorithm
    random_state: RandomState,
    /// Partitioning mode to use
    pub mode: PartitionMode,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// The projection indices of the columns in the output schema of join
    pub projection: Option<Vec<usize>>,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Null matching behavior: If `null_equals_null` is true, rows that have
    /// `null`s in both left and right equijoin columns will be matched.
    /// Otherwise, rows that have `null`s in the join columns will not be
    /// matched and thus will not appear in the output.
    pub null_equals_null: bool,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    // cache: PlanProperties,
    /// Hash table corresponding to the build side
    join_data: JoinLeftData,
}

impl HashProbeExec {
    /// Tries to create a new [HashProbeExec].
    ///
    /// # Error
    /// This function errors when it is not possible to join the left and right sides on keys `on`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
        partition_mode: PartitionMode,
        null_equals_null: bool,
        join_data: JoinLeftData,
    ) -> Result<Self> {
        let left_schema = join_data.batch.schema();
        let right_schema = right.schema();
        if on.is_empty() {
            return plan_err!("On constraints in HashProbeExec should be non-empty");
        }

        check_join_is_valid(&left_schema, &right_schema, &on)?;

        let (join_schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, join_type);

        let random_state = RandomState::with_seeds(0, 0, 0, 0);

        let join_schema = Arc::new(join_schema);

        //  check if the projection is valid
        can_project(&join_schema, projection.as_ref())?;

        // let cache = Self::compute_properties(
        //     &right,
        //     join_schema.clone(),
        //     *join_type,
        //     &on,
        //     partition_mode,
        //     projection.as_ref(),
        // )?;

        Ok(HashProbeExec {
            right,
            on,
            filter,
            join_type: *join_type,
            join_schema,
            left_fut: Default::default(),
            random_state,
            mode: partition_mode,
            metrics: ExecutionPlanMetricsSet::new(),
            projection,
            column_indices,
            null_equals_null,
            // cache,
            join_data,
        })
    }

    /// right (probe) side which are filtered by the hash table
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Set of common columns used to join on
    pub fn on(&self) -> &[(PhysicalExprRef, PhysicalExprRef)] {
        &self.on
    }

    /// Filters applied before join output
    pub fn filter(&self) -> Option<&JoinFilter> {
        self.filter.as_ref()
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// The partitioning mode of this hash join
    pub fn partition_mode(&self) -> &PartitionMode {
        &self.mode
    }

    /// Get null_equals_null
    pub fn null_equals_null(&self) -> bool {
        self.null_equals_null
    }

    /// Calculate order preservation flags for this hash join.
    fn maintains_input_order(join_type: JoinType) -> Vec<bool> {
        vec![
            false,
            matches!(
                join_type,
                JoinType::Inner | JoinType::RightAnti | JoinType::RightSemi
            ),
        ]
    }

    /// Get probe side information for the hash join.
    pub fn probe_side() -> JoinSide {
        // In current implementation right side is always probe side.
        JoinSide::Right
    }

    /// Return whether the join contains a projection
    pub fn contain_projection(&self) -> bool {
        self.projection.is_some()
    }

    /// Return new instance of [HashProbeExec] with the given projection.
    pub fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        //  check if the projection is valid
        can_project(&self.schema(), projection.as_ref())?;
        let projection = match projection {
            Some(projection) => match &self.projection {
                Some(p) => Some(projection.iter().map(|i| p[*i]).collect()),
                None => Some(projection),
            },
            None => None,
        };
        Self::try_new(
            self.right.clone(),
            self.on.clone(),
            self.filter.clone(),
            &self.join_type,
            projection,
            self.mode,
            self.null_equals_null,
            self.join_data.clone(),
        )
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        right: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        join_type: JoinType,
        on: JoinOnRef,
        mode: PartitionMode,
        projection: Option<&Vec<usize>>,
    ) -> Result<PlanProperties> {
        todo!();
        // Calculate equivalence properties:
        // let mut eq_properties = join_equivalence_properties(
        //     left.equivalence_properties().clone(),
        //     right.equivalence_properties().clone(),
        //     &join_type,
        //     schema.clone(),
        //     &Self::maintains_input_order(join_type),
        //     Some(Self::probe_side()),
        //     on,
        // );

        // // Get output partitioning:
        // let left_columns_len = left.schema().fields.len();
        // let mut output_partitioning = match mode {
        //     PartitionMode::CollectLeft => match join_type {
        //         JoinType::Inner | JoinType::Right => adjust_right_output_partitioning(
        //             right.output_partitioning(),
        //             left_columns_len,
        //         ),
        //         JoinType::RightSemi | JoinType::RightAnti => {
        //             right.output_partitioning().clone()
        //         }
        //         JoinType::Left
        //         | JoinType::LeftSemi
        //         | JoinType::LeftAnti
        //         | JoinType::Full => Partitioning::UnknownPartitioning(
        //             right.output_partitioning().partition_count(),
        //         ),
        //     },
        //     PartitionMode::Partitioned => partitioned_join_output_partitioning(
        //         join_type,
        //         left.output_partitioning(),
        //         right.output_partitioning(),
        //         left_columns_len,
        //     ),
        //     PartitionMode::Auto => Partitioning::UnknownPartitioning(
        //         right.output_partitioning().partition_count(),
        //     ),
        // };

        // // Determine execution mode by checking whether this join is pipeline
        // // breaking. This happens when the left side is unbounded, or the right
        // // side is unbounded with `Left`, `Full`, `LeftAnti` or `LeftSemi` join types.
        // let pipeline_breaking = left.execution_mode().is_unbounded()
        //     || (right.execution_mode().is_unbounded()
        //         && matches!(
        //             join_type,
        //             JoinType::Left
        //                 | JoinType::Full
        //                 | JoinType::LeftAnti
        //                 | JoinType::LeftSemi
        //         ));

        // let mode = if pipeline_breaking {
        //     ExecutionMode::PipelineBreaking
        // } else {
        //     execution_mode_from_children([left, right])
        // };

        // // If contains projection, update the PlanProperties.
        // if let Some(projection) = projection {
        //     let projection_exprs = project_index_to_exprs(projection, &schema);
        //     // construct a map from the input expressions to the output expression of the Projection
        //     let projection_mapping =
        //         ProjectionMapping::try_new(&projection_exprs, &schema)?;
        //     let out_schema = project_schema(&schema, Some(projection))?;
        //     if let Partitioning::Hash(exprs, part) = output_partitioning {
        //         let normalized_exprs = exprs
        //             .iter()
        //             .map(|expr| {
        //                 eq_properties
        //                     .project_expr(expr, &projection_mapping)
        //                     .unwrap_or_else(|| {
        //                         Arc::new(UnKnownColumn::new(&expr.to_string()))
        //                     })
        //             })
        //             .collect();
        //         output_partitioning = Partitioning::Hash(normalized_exprs, part);
        //     }
        //     eq_properties = eq_properties.project(&projection_mapping, out_schema);
        // }
        // Ok(PlanProperties::new(
        //     eq_properties,
        //     output_partitioning,
        //     mode,
        // ))
    }
}

impl DisplayAs for HashProbeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let display_filter = self.filter.as_ref().map_or_else(
                    || "".to_string(),
                    |f| format!(", filter={}", f.expression()),
                );
                let display_projections = if self.contain_projection() {
                    format!(
                        ", projection=[{}]",
                        self.projection
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|index| format!(
                                "{}@{}",
                                self.join_schema.fields().get(*index).unwrap().name(),
                                index
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    "".to_string()
                };
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({}, {})", c1, c2))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(
                    f,
                    "HashProbeExec: mode={:?}, join_type={:?}, on=[{}]{}{}",
                    self.mode, self.join_type, on, display_filter, display_projections
                )
            }
        }
    }
}

fn project_index_to_exprs(
    projection_index: &[usize],
    schema: &SchemaRef,
) -> Vec<(Arc<dyn PhysicalExpr>, String)> {
    projection_index
        .iter()
        .map(|index| {
            let field = schema.field(*index);
            (
                Arc::new(datafusion_physical_expr::expressions::Column::new(
                    field.name(),
                    *index,
                )) as Arc<dyn PhysicalExpr>,
                field.name().to_owned(),
            )
        })
        .collect::<Vec<_>>()
}

impl ExecutionPlan for HashProbeExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        todo!()
        // &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        match self.mode {
            PartitionMode::CollectLeft => vec![
                Distribution::SinglePartition,
                Distribution::UnspecifiedDistribution,
            ],
            PartitionMode::Partitioned => {
                let (left_expr, right_expr) =
                    self.on.iter().map(|(l, r)| (l.clone(), r.clone())).unzip();
                vec![
                    Distribution::HashPartitioned(left_expr),
                    Distribution::HashPartitioned(right_expr),
                ]
            }
            PartitionMode::Auto => vec![
                Distribution::UnspecifiedDistribution,
                Distribution::UnspecifiedDistribution,
            ],
        }
    }

    // For [JoinType::Inner] and [JoinType::RightSemi] in hash joins, the probe phase initiates by
    // applying the hash function to convert the join key(s) in each row into a hash value from the
    // probe side table in the order they're arranged. The hash value is used to look up corresponding
    // entries in the hash table that was constructed from the build side table during the build phase.
    //
    // Because of the immediate generation of result rows once a match is found,
    // the output of the join tends to follow the order in which the rows were read from
    // the probe side table. This is simply due to the sequence in which the rows were processed.
    // Hence, it appears that the hash join is preserving the order of the probe side.
    //
    // Meanwhile, in the case of a [JoinType::RightAnti] hash join,
    // the unmatched rows from the probe side are also kept in order.
    // This is because the **`RightAnti`** join is designed to return rows from the right
    // (probe side) table that have no match in the left (build side) table. Because the rows
    // are processed sequentially in the probe phase, and unmatched rows are directly output
    // as results, these results tend to retain the order of the probe side table.
    fn maintains_input_order(&self) -> Vec<bool> {
        Self::maintains_input_order(self.join_type)
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.right.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match &children[..] {
            [right] => Ok(Arc::new(HashProbeExec::try_new(
                right.clone(),
                self.on.clone(),
                self.filter.clone(),
                &self.join_type,
                self.projection.clone(),
                self.mode,
                self.null_equals_null,
                self.join_data.clone(),
            )?)),
            _ => internal_err!("SortMergeJoin wrong number of children"),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let on_left = self.on.iter().map(|on| on.0.clone()).collect::<Vec<_>>();
        let on_right = self.on.iter().map(|on| on.1.clone()).collect::<Vec<_>>();
        let left_partitions = self.join_data.output_partitioning.partition_count();
        let right_partitions = self.right.output_partitioning().partition_count();

        if self.mode == PartitionMode::Partitioned && left_partitions != right_partitions
        {
            return internal_err!(
                "Invalid HashProbeExec, partition count mismatch {left_partitions}!={right_partitions},\
                 consider using RepartitionExec"
            );
        }

        let join_metrics = BuildProbeJoinMetrics::new(partition, &self.metrics);

        let batch_size = context.session_config().batch_size();

        let reservation = MemoryConsumer::new(format!("HashJoinStream[{partition}]"))
            .register(context.memory_pool());

        // we have the batches and the hash map with their keys. We can how create a stream
        // over the right that uses this information to issue new batches.
        let right_stream = self.right.execute(partition, context)?;

        // update column indices to reflect the projection
        let column_indices_after_projection = match &self.projection {
            Some(projection) => projection
                .iter()
                .map(|i| self.column_indices[*i].clone())
                .collect(),
            None => self.column_indices.clone(),
        };

        // Reserving memory for visited_left_side bitmap in case it hasn't been initialized yet
        // and join_type requires to store it
        if need_produce_result_in_final(self.join_type) {
            // TODO: Replace `ceil` wrapper with stable `div_cell` after
            // https://github.com/rust-lang/rust/issues/88581
            let visited_bitmap_size = bit_util::ceil(self.join_data.num_rows(), 8);
            // self.reservation.try_grow(visited_bitmap_size)?;
            // self.join_metrics.build_mem_used.add(visited_bitmap_size);
        }

        let visited_left_side = if need_produce_result_in_final(self.join_type) {
            let num_rows = self.join_data.num_rows();
            // Some join types need to track which row has be matched or unmatched:
            // `left semi` join:  need to use the bitmap to produce the matched row in the left side
            // `left` join:       need to use the bitmap to produce the unmatched row in the left side with null
            // `left anti` join:  need to use the bitmap to produce the unmatched row in the left side
            // `full` join:       need to use the bitmap to produce the unmatched row in the left side with null
            let mut buffer = BooleanBufferBuilder::new(num_rows);
            buffer.append_n(num_rows, false);
            buffer
        } else {
            BooleanBufferBuilder::new(0)
        };

        let build_side = BuildSide::Ready(BuildSideReadyState {
            left_data: Arc::new(self.join_data.clone()),
            visited_left_side,
        });
        // set initial state to fetch probe batch to indicate that we do not
        // need to build the hash table
        Ok(Box::pin(HashJoinStream {
            schema: self.schema(),
            on_left,
            on_right,
            filter: self.filter.clone(),
            join_type: self.join_type,
            right: right_stream,
            column_indices: column_indices_after_projection,
            random_state: self.random_state.clone(),
            join_metrics,
            null_equals_null: self.null_equals_null,
            reservation,
            state: HashJoinStreamState::FetchProbeBatch,
            build_side,
            batch_size,
            hashes_buffer: vec![],
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        // TODO stats: it is not possible in general to know the output size of joins
        // There are some special cases though, for example:
        // - `A LEFT JOIN B ON A.col=B.col` with `COUNT_DISTINCT(B.col)=COUNT(B.col)`
        todo!()
        // let mut stats = estimate_join_statistics(
        //     self.left.clone(),
        //     self.right.clone(),
        //     self.on.clone(),
        //     &self.join_type,
        //     &self.join_schema,
        // )?;
        // // Project statistics if there is a projection
        // if let Some(projection) = &self.projection {
        //     stats.column_statistics = stats
        //         .column_statistics
        //         .into_iter()
        //         .enumerate()
        //         .filter(|(i, _)| projection.contains(i))
        //         .map(|(_, s)| s)
        //         .collect();
        // }
        // Ok(stats)
    }
}

/// Updates `hash_map` with new entries from `batch` evaluated against the expressions `on`
/// using `offset` as a start value for `batch` row indices.
///
/// `fifo_hashmap` sets the order of iteration over `batch` rows while updating hashmap,
/// which allows to keep either first (if set to true) or last (if set to false) row index
/// as a chain head for rows with equal hash values.
#[allow(clippy::too_many_arguments)]
pub fn update_hash<T>(
    on: &[PhysicalExprRef],
    batch: &RecordBatch,
    hash_map: &mut T,
    offset: usize,
    random_state: &RandomState,
    hashes_buffer: &mut Vec<u64>,
    deleted_offset: usize,
    fifo_hashmap: bool,
) -> Result<()>
where
    T: JoinHashMapType,
{
    // evaluate the keys
    let keys_values = on
        .iter()
        .map(|c| c.evaluate(batch)?.into_array(batch.num_rows()))
        .collect::<Result<Vec<_>>>()?;

    // calculate the hash values
    let hash_values = create_hashes(&keys_values, random_state, hashes_buffer)?;

    // For usual JoinHashmap, the implementation is void.
    hash_map.extend_zero(batch.num_rows());

    // Updating JoinHashMap from hash values iterator
    let hash_values_iter = hash_values
        .iter()
        .enumerate()
        .map(|(i, val)| (i + offset, val));

    if fifo_hashmap {
        hash_map.update_from_iter(hash_values_iter.rev(), deleted_offset);
    } else {
        hash_map.update_from_iter(hash_values_iter, deleted_offset);
    }

    Ok(())
}

/// Represents build-side of hash join.
enum BuildSide {
    /// Indicates that build-side not collected yet
    Initial(BuildSideInitialState),
    /// Indicates that build-side data has been collected
    Ready(BuildSideReadyState),
}

/// Container for BuildSide::Initial related data
struct BuildSideInitialState {
    /// Future for building hash table from build-side input
    left_fut: OnceFut<JoinLeftData>,
}

/// Container for BuildSide::Ready related data
struct BuildSideReadyState {
    /// Collected build-side data
    left_data: Arc<JoinLeftData>,
    /// Which build-side rows have been matched while creating output.
    /// For some OUTER joins, we need to know which rows have not been matched
    /// to produce the correct output.
    visited_left_side: BooleanBufferBuilder,
}

impl BuildSide {
    /// Tries to extract BuildSideInitialState from BuildSide enum.
    /// Returns an error if state is not Initial.
    fn try_as_initial_mut(&mut self) -> Result<&mut BuildSideInitialState> {
        match self {
            BuildSide::Initial(state) => Ok(state),
            _ => internal_err!("Expected build side in initial state"),
        }
    }

    /// Tries to extract BuildSideReadyState from BuildSide enum.
    /// Returns an error if state is not Ready.
    fn try_as_ready(&self) -> Result<&BuildSideReadyState> {
        match self {
            BuildSide::Ready(state) => Ok(state),
            _ => internal_err!("Expected build side in ready state"),
        }
    }

    /// Tries to extract BuildSideReadyState from BuildSide enum.
    /// Returns an error if state is not Ready.
    fn try_as_ready_mut(&mut self) -> Result<&mut BuildSideReadyState> {
        match self {
            BuildSide::Ready(state) => Ok(state),
            _ => internal_err!("Expected build side in ready state"),
        }
    }
}

/// Represents state of HashJoinStream
///
/// Expected state transitions performed by HashJoinStream are:
///
/// ```text
///
///       WaitBuildSide
///             │
///             ▼
///  ┌─► FetchProbeBatch ───► ExhaustedProbeSide ───► Completed
///  │          │
///  │          ▼
///  └─ ProcessProbeBatch
///
/// ```
enum HashJoinStreamState {
    /// Initial state for HashJoinStream indicating that build-side data not collected yet
    WaitBuildSide,
    /// Indicates that build-side has been collected, and stream is ready for fetching probe-side
    FetchProbeBatch,
    /// Indicates that non-empty batch has been fetched from probe-side, and is ready to be processed
    ProcessProbeBatch(ProcessProbeBatchState),
    /// Indicates that probe-side has been fully processed
    ExhaustedProbeSide,
    /// Indicates that HashJoinStream execution is completed
    Completed,
}

impl HashJoinStreamState {
    /// Tries to extract ProcessProbeBatchState from HashJoinStreamState enum.
    /// Returns an error if state is not ProcessProbeBatchState.
    fn try_as_process_probe_batch_mut(&mut self) -> Result<&mut ProcessProbeBatchState> {
        match self {
            HashJoinStreamState::ProcessProbeBatch(state) => Ok(state),
            _ => internal_err!("Expected hash join stream in ProcessProbeBatch state"),
        }
    }
}

/// Container for HashJoinStreamState::ProcessProbeBatch related data
struct ProcessProbeBatchState {
    /// Current probe-side batch
    batch: RecordBatch,
    /// Starting offset for JoinHashMap lookups
    offset: JoinHashMapOffset,
    /// Max joined probe-side index from current batch
    joined_probe_idx: Option<usize>,
}

impl ProcessProbeBatchState {
    fn advance(&mut self, offset: JoinHashMapOffset, joined_probe_idx: Option<usize>) {
        self.offset = offset;
        if joined_probe_idx.is_some() {
            self.joined_probe_idx = joined_probe_idx;
        }
    }
}

/// [`Stream`] for [`HashProbeExec`] that does the actual join.
///
/// This stream:
///
/// 1. Reads the entire left input (build) and constructs a hash table
///
/// 2. Streams [RecordBatch]es as they arrive from the right input (probe) and joins
/// them with the contents of the hash table
struct HashJoinStream {
    /// Input schema
    schema: Arc<Schema>,
    /// equijoin columns from the left (build side)
    on_left: Vec<PhysicalExprRef>,
    /// equijoin columns from the right (probe side)
    on_right: Vec<PhysicalExprRef>,
    /// optional join filter
    filter: Option<JoinFilter>,
    /// type of the join (left, right, semi, etc)
    join_type: JoinType,
    /// right (probe) input
    right: SendableRecordBatchStream,
    /// Random state used for hashing initialization
    random_state: RandomState,
    /// Metrics
    join_metrics: BuildProbeJoinMetrics,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// If null_equals_null is true, null == null else null != null
    null_equals_null: bool,
    /// Memory reservation
    reservation: MemoryReservation,
    /// State of the stream
    state: HashJoinStreamState,
    /// Build side
    build_side: BuildSide,
    /// Maximum output batch size
    batch_size: usize,
    /// Scratch space for computing hashes
    hashes_buffer: Vec<u64>,
}

impl RecordBatchStream for HashJoinStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Executes lookups by hash against JoinHashMap and resolves potential
/// hash collisions.
/// Returns build/probe indices satisfying the equality condition, along with
/// (optional) starting point for next iteration.
///
/// # Example
///
/// For `LEFT.b1 = RIGHT.b2`:
/// LEFT (build) Table:
/// ```text
///  a1  b1  c1
///  1   1   10
///  3   3   30
///  5   5   50
///  7   7   70
///  9   8   90
///  11  8   110
///  13   10  130
/// ```
///
/// RIGHT (probe) Table:
/// ```text
///  a2   b2  c2
///  2    2   20
///  4    4   40
///  6    6   60
///  8    8   80
/// 10   10  100
/// 12   10  120
/// ```
///
/// The result is
/// ```text
/// "+----+----+-----+----+----+-----+",
/// "| a1 | b1 | c1  | a2 | b2 | c2  |",
/// "+----+----+-----+----+----+-----+",
/// "| 9  | 8  | 90  | 8  | 8  | 80  |",
/// "| 11 | 8  | 110 | 8  | 8  | 80  |",
/// "| 13 | 10 | 130 | 10 | 10 | 100 |",
/// "| 13 | 10 | 130 | 12 | 10 | 120 |",
/// "+----+----+-----+----+----+-----+"
/// ```
///
/// And the result of build and probe indices are:
/// ```text
/// Build indices: 4, 5, 6, 6
/// Probe indices: 3, 3, 4, 5
/// ```
#[allow(clippy::too_many_arguments)]
fn lookup_join_hashmap(
    build_hashmap: &JoinHashMap,
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_on: &[PhysicalExprRef],
    probe_on: &[PhysicalExprRef],
    null_equals_null: bool,
    hashes_buffer: &[u64],
    limit: usize,
    offset: JoinHashMapOffset,
) -> Result<(UInt64Array, UInt32Array, Option<JoinHashMapOffset>)> {
    let keys_values = probe_on
        .iter()
        .map(|c| c.evaluate(probe_batch)?.into_array(probe_batch.num_rows()))
        .collect::<Result<Vec<_>>>()?;
    let build_join_values = build_on
        .iter()
        .map(|c| {
            c.evaluate(build_input_buffer)?
                .into_array(build_input_buffer.num_rows())
        })
        .collect::<Result<Vec<_>>>()?;

    let (mut probe_builder, mut build_builder, next_offset) = build_hashmap
        .get_matched_indices_with_limit_offset(hashes_buffer, None, limit, offset);

    let build_indices: UInt64Array =
        PrimitiveArray::new(build_builder.finish().into(), None);
    let probe_indices: UInt32Array =
        PrimitiveArray::new(probe_builder.finish().into(), None);

    let (build_indices, probe_indices) = equal_rows_arr(
        &build_indices,
        &probe_indices,
        &build_join_values,
        &keys_values,
        null_equals_null,
    )?;

    Ok((build_indices, probe_indices, next_offset))
}

// version of eq_dyn supporting equality on null arrays
fn eq_dyn_null(
    left: &dyn Array,
    right: &dyn Array,
    null_equals_null: bool,
) -> Result<BooleanArray, ArrowError> {
    match (left.data_type(), right.data_type()) {
        _ if null_equals_null => not_distinct(&left, &right),
        _ => eq(&left, &right),
    }
}

pub fn equal_rows_arr(
    indices_left: &UInt64Array,
    indices_right: &UInt32Array,
    left_arrays: &[ArrayRef],
    right_arrays: &[ArrayRef],
    null_equals_null: bool,
) -> Result<(UInt64Array, UInt32Array)> {
    let mut iter = left_arrays.iter().zip(right_arrays.iter());

    let (first_left, first_right) = iter.next().ok_or_else(|| {
        DataFusionError::Internal(
            "At least one array should be provided for both left and right".to_string(),
        )
    })?;

    let arr_left = take(first_left.as_ref(), indices_left, None)?;
    let arr_right = take(first_right.as_ref(), indices_right, None)?;

    let mut equal: BooleanArray = eq_dyn_null(&arr_left, &arr_right, null_equals_null)?;

    // Use map and try_fold to iterate over the remaining pairs of arrays.
    // In each iteration, take is used on the pair of arrays and their equality is determined.
    // The results are then folded (combined) using the and function to get a final equality result.
    equal = iter
        .map(|(left, right)| {
            let arr_left = take(left.as_ref(), indices_left, None)?;
            let arr_right = take(right.as_ref(), indices_right, None)?;
            eq_dyn_null(arr_left.as_ref(), arr_right.as_ref(), null_equals_null)
        })
        .try_fold(equal, |acc, equal2| and(&acc, &equal2?))?;

    let filter_builder = FilterBuilder::new(&equal).optimize().build();

    let left_filtered = filter_builder.filter(indices_left)?;
    let right_filtered = filter_builder.filter(indices_right)?;

    Ok((
        downcast_array(left_filtered.as_ref()),
        downcast_array(right_filtered.as_ref()),
    ))
}

impl HashJoinStream {
    /// Separate implementation function that unpins the [`HashJoinStream`] so
    /// that partial borrows work correctly
    fn poll_next_impl(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            return match self.state {
                HashJoinStreamState::WaitBuildSide => {
                    handle_state!(ready!(self.collect_build_side(cx)))
                }
                HashJoinStreamState::FetchProbeBatch => {
                    handle_state!(ready!(self.fetch_probe_batch(cx)))
                }
                HashJoinStreamState::ProcessProbeBatch(_) => {
                    handle_state!(self.process_probe_batch())
                }
                HashJoinStreamState::ExhaustedProbeSide => {
                    handle_state!(self.process_unmatched_build_batch())
                }
                HashJoinStreamState::Completed => Poll::Ready(None),
            };
        }
    }

    /// Collects build-side data by polling `OnceFut` future from initialized build-side
    ///
    /// Updates build-side to `Ready`, and state to `FetchProbeSide`
    fn collect_build_side(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let build_timer = self.join_metrics.build_time.timer();
        // build hash table from left (build) side, if not yet done
        let left_data = ready!(self
            .build_side
            .try_as_initial_mut()?
            .left_fut
            .get_shared(cx))?;
        build_timer.done();

        // Reserving memory for visited_left_side bitmap in case it hasn't been initialized yet
        // and join_type requires to store it
        if need_produce_result_in_final(self.join_type) {
            // TODO: Replace `ceil` wrapper with stable `div_cell` after
            // https://github.com/rust-lang/rust/issues/88581
            let visited_bitmap_size = bit_util::ceil(left_data.num_rows(), 8);
            self.reservation.try_grow(visited_bitmap_size)?;
            self.join_metrics.build_mem_used.add(visited_bitmap_size);
        }

        let visited_left_side = if need_produce_result_in_final(self.join_type) {
            let num_rows = left_data.num_rows();
            // Some join types need to track which row has be matched or unmatched:
            // `left semi` join:  need to use the bitmap to produce the matched row in the left side
            // `left` join:       need to use the bitmap to produce the unmatched row in the left side with null
            // `left anti` join:  need to use the bitmap to produce the unmatched row in the left side
            // `full` join:       need to use the bitmap to produce the unmatched row in the left side with null
            let mut buffer = BooleanBufferBuilder::new(num_rows);
            buffer.append_n(num_rows, false);
            buffer
        } else {
            BooleanBufferBuilder::new(0)
        };

        self.state = HashJoinStreamState::FetchProbeBatch;
        self.build_side = BuildSide::Ready(BuildSideReadyState {
            left_data,
            visited_left_side,
        });

        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }

    /// Fetches next batch from probe-side
    ///
    /// If non-empty batch has been fetched, updates state to `ProcessProbeBatchState`,
    /// otherwise updates state to `ExhaustedProbeSide`
    fn fetch_probe_batch(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        match ready!(self.right.poll_next_unpin(cx)) {
            None => {
                self.state = HashJoinStreamState::ExhaustedProbeSide;
            }
            Some(Ok(batch)) => {
                // Precalculate hash values for fetched batch
                let keys_values = self
                    .on_right
                    .iter()
                    .map(|c| c.evaluate(&batch)?.into_array(batch.num_rows()))
                    .collect::<Result<Vec<_>>>()?;

                self.hashes_buffer.clear();
                self.hashes_buffer.resize(batch.num_rows(), 0);
                create_hashes(&keys_values, &self.random_state, &mut self.hashes_buffer)?;

                self.join_metrics.input_batches.add(1);
                self.join_metrics.input_rows.add(batch.num_rows());

                self.state =
                    HashJoinStreamState::ProcessProbeBatch(ProcessProbeBatchState {
                        batch,
                        offset: (0, None),
                        joined_probe_idx: None,
                    });
            }
            Some(Err(err)) => return Poll::Ready(Err(err)),
        };

        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }

    /// Joins current probe batch with build-side data and produces batch with matched output
    ///
    /// Updates state to `FetchProbeBatch`
    fn process_probe_batch(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        let state = self.state.try_as_process_probe_batch_mut()?;
        let build_side = self.build_side.try_as_ready_mut()?;

        let timer = self.join_metrics.join_time.timer();

        // get the matched by join keys indices
        let (left_indices, right_indices, next_offset) = lookup_join_hashmap(
            build_side.left_data.hash_map(),
            build_side.left_data.batch(),
            &state.batch,
            &self.on_left,
            &self.on_right,
            self.null_equals_null,
            &self.hashes_buffer,
            self.batch_size,
            state.offset,
        )?;

        // apply join filter if exists
        let (left_indices, right_indices) = if let Some(filter) = &self.filter {
            apply_join_filter_to_indices(
                build_side.left_data.batch(),
                &state.batch,
                left_indices,
                right_indices,
                filter,
                JoinSide::Left,
            )?
        } else {
            (left_indices, right_indices)
        };

        // mark joined left-side indices as visited, if required by join type
        if need_produce_result_in_final(self.join_type) {
            left_indices.iter().flatten().for_each(|x| {
                build_side.visited_left_side.set_bit(x as usize, true);
            });
        }

        // The goals of index alignment for different join types are:
        //
        // 1) Right & FullJoin -- to append all missing probe-side indices between
        //    previous (excluding) and current joined indices.
        // 2) SemiJoin -- deduplicate probe indices in range between previous
        //    (excluding) and current joined indices.
        // 3) AntiJoin -- return only missing indices in range between
        //    previous and current joined indices.
        //    Inclusion/exclusion of the indices themselves don't matter
        //
        // As a summary -- alignment range can be produced based only on
        // joined (matched with filters applied) probe side indices, excluding starting one
        // (left from previous iteration).

        // if any rows have been joined -- get last joined probe-side (right) row
        // it's important that index counts as "joined" after hash collisions checks
        // and join filters applied.
        let last_joined_right_idx = match right_indices.len() {
            0 => None,
            n => Some(right_indices.value(n - 1) as usize),
        };

        // Calculate range and perform alignment.
        // In case probe batch has been processed -- align all remaining rows.
        let index_alignment_range_start = state.joined_probe_idx.map_or(0, |v| v + 1);
        let index_alignment_range_end = if next_offset.is_none() {
            state.batch.num_rows()
        } else {
            last_joined_right_idx.map_or(0, |v| v + 1)
        };

        let (left_indices, right_indices) = adjust_indices_by_join_type(
            left_indices,
            right_indices,
            index_alignment_range_start..index_alignment_range_end,
            self.join_type,
        );

        let result = build_batch_from_indices(
            &self.schema,
            build_side.left_data.batch(),
            &state.batch,
            &left_indices,
            &right_indices,
            &self.column_indices,
            JoinSide::Left,
        )?;

        self.join_metrics.output_batches.add(1);
        self.join_metrics.output_rows.add(result.num_rows());
        timer.done();

        if next_offset.is_none() {
            self.state = HashJoinStreamState::FetchProbeBatch;
        } else {
            state.advance(
                next_offset
                    .ok_or_else(|| internal_datafusion_err!("unexpected None offset"))?,
                last_joined_right_idx,
            )
        };

        Ok(StatefulStreamResult::Ready(Some(result)))
    }

    /// Processes unmatched build-side rows for certain join types and produces output batch
    ///
    /// Updates state to `Completed`
    fn process_unmatched_build_batch(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        let timer = self.join_metrics.join_time.timer();

        if !need_produce_result_in_final(self.join_type) {
            self.state = HashJoinStreamState::Completed;

            return Ok(StatefulStreamResult::Continue);
        }

        let build_side = self.build_side.try_as_ready()?;

        // use the global left bitmap to produce the left indices and right indices
        let (left_side, right_side) =
            get_final_indices_from_bit_map(&build_side.visited_left_side, self.join_type);
        let empty_right_batch = RecordBatch::new_empty(self.right.schema());
        // use the left and right indices to produce the batch result
        let result = build_batch_from_indices(
            &self.schema,
            build_side.left_data.batch(),
            &empty_right_batch,
            &left_side,
            &right_side,
            &self.column_indices,
            JoinSide::Left,
        );

        if let Ok(ref batch) = result {
            self.join_metrics.input_batches.add(1);
            self.join_metrics.input_rows.add(batch.num_rows());

            self.join_metrics.output_batches.add(1);
            self.join_metrics.output_rows.add(batch.num_rows());
        }
        timer.done();

        self.state = HashJoinStreamState::Completed;

        Ok(StatefulStreamResult::Ready(Some(result?)))
    }
}

impl Stream for HashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}
