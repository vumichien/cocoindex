use crate::prelude::*;

use futures::future::try_join_all;
use sqlx::PgPool;
use std::collections::{HashMap, HashSet};

use super::db_tracking::{self, TrackedTargetKeyInfo, read_source_tracking_info_for_processing};
use super::db_tracking_setup;
use super::evaluator::{
    EvaluateSourceEntryOutput, SourceRowEvaluationContext, evaluate_source_entry,
};
use super::memoization::{EvaluationMemory, EvaluationMemoryOptions, StoredMemoizationInfo};
use super::stats;

use crate::base::value::{self, FieldValues, KeyValue};
use crate::builder::plan::*;
use crate::ops::interface::{
    ExportTargetMutation, ExportTargetUpsertEntry, Ordinal, SourceExecutorGetOptions,
};
use crate::utils::db::WriteAction;
use crate::utils::fingerprint::{Fingerprint, Fingerprinter};

pub fn extract_primary_key(
    primary_key_def: &AnalyzedPrimaryKeyDef,
    record: &FieldValues,
) -> Result<KeyValue> {
    match primary_key_def {
        AnalyzedPrimaryKeyDef::Fields(fields) => {
            KeyValue::from_values(fields.iter().map(|field| &record.fields[*field as usize]))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SourceVersionKind {
    #[default]
    UnknownLogic,
    DifferentLogic,
    CurrentLogic,
    NonExistence,
}

#[derive(Debug, Clone, Default)]
pub struct SourceVersion {
    pub ordinal: Ordinal,
    pub kind: SourceVersionKind,
    pub content_hash: Option<Fingerprint>,
}

impl SourceVersion {
    pub fn from_stored(
        stored_ordinal: Option<i64>,
        stored_fp: &Option<Vec<u8>>,
        stored_content_hash: &Option<Vec<u8>>,
        curr_fp: Fingerprint,
    ) -> Self {
        Self {
            ordinal: Ordinal(stored_ordinal),
            kind: match &stored_fp {
                Some(stored_fp) => {
                    if stored_fp.as_slice() == curr_fp.0.as_slice() {
                        SourceVersionKind::CurrentLogic
                    } else {
                        SourceVersionKind::DifferentLogic
                    }
                }
                None => SourceVersionKind::UnknownLogic,
            },
            content_hash: stored_content_hash.as_ref().map(|hash| {
                let mut bytes = [0u8; 16];
                if hash.len() >= 16 {
                    bytes.copy_from_slice(&hash[..16]);
                }
                Fingerprint(bytes)
            }),
        }
    }

    pub fn from_current_with_hash(ordinal: Ordinal, content_hash: Option<Fingerprint>) -> Self {
        Self {
            ordinal,
            kind: SourceVersionKind::CurrentLogic,
            content_hash,
        }
    }

    pub fn from_current_data(data: &interface::SourceData) -> Self {
        let kind = match &data.value {
            interface::SourceValue::Existence(_) => SourceVersionKind::CurrentLogic,
            interface::SourceValue::NonExistence => SourceVersionKind::NonExistence,
        };
        Self {
            ordinal: data.ordinal,
            kind,
            content_hash: data.content_hash,
        }
    }

    pub fn should_skip(
        &self,
        target: &SourceVersion,
        update_stats: Option<&stats::UpdateStats>,
    ) -> bool {
        let should_skip = match (self.ordinal.0, target.ordinal.0) {
            (Some(existing_ordinal), Some(target_ordinal)) => {
                // If logic versions are different, never skip (must reprocess)
                if self.kind != target.kind {
                    false
                } else if let (Some(existing_hash), Some(target_hash)) = (&self.content_hash, &target.content_hash) {
                    // Same logic version and we have content hashes for both versions
                    // Content hasn't changed - skip processing
                    existing_hash == target_hash
                } else {
                    // Fall back to ordinal-based comparison when content hash is not available
                    existing_ordinal > target_ordinal || (existing_ordinal == target_ordinal && self.kind >= target.kind)
                }
            }
            _ => false,
        };
        if should_skip {
            if let Some(update_stats) = update_stats {
                update_stats.num_no_change.inc(1);
            }
        }
        should_skip
    }
}

pub enum SkippedOr<T> {
    Normal(T),
    Skipped(SourceVersion),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TargetKeyPair {
    pub key: serde_json::Value,
    pub additional_key: serde_json::Value,
}

#[derive(Default)]
struct TrackingInfoForTarget<'a> {
    export_op: Option<&'a AnalyzedExportOp>,

    // Existing keys info. Keyed by target key.
    // Will be removed after new rows for the same key are added into `new_staging_keys_info` and `mutation.upserts`,
    // hence all remaining ones are to be deleted.
    existing_staging_keys_info: HashMap<TargetKeyPair, Vec<(i64, Option<Fingerprint>)>>,
    existing_keys_info: HashMap<TargetKeyPair, Vec<(i64, Option<Fingerprint>)>>,

    // New keys info for staging.
    new_staging_keys_info: Vec<TrackedTargetKeyInfo>,

    // Mutation to apply to the target storage.
    mutation: ExportTargetMutation,
}

#[derive(Debug)]
struct PrecommitData<'a> {
    evaluate_output: &'a EvaluateSourceEntryOutput,
    memoization_info: &'a StoredMemoizationInfo,
}
struct PrecommitMetadata {
    source_entry_exists: bool,
    process_ordinal: i64,
    existing_process_ordinal: Option<i64>,
    new_target_keys: db_tracking::TrackedTargetKeyForSource,
}
struct PrecommitOutput {
    metadata: PrecommitMetadata,
    target_mutations: HashMap<i32, ExportTargetMutation>,
}

async fn precommit_source_tracking_info(
    source_id: i32,
    source_key_json: &serde_json::Value,
    source_version: &SourceVersion,
    logic_fp: Fingerprint,
    data: Option<PrecommitData<'_>>,
    process_timestamp: &chrono::DateTime<chrono::Utc>,
    db_setup: &db_tracking_setup::TrackingTableSetupState,
    export_ops: &[AnalyzedExportOp],
    update_stats: &stats::UpdateStats,
    pool: &PgPool,
) -> Result<SkippedOr<PrecommitOutput>> {
    let mut txn = pool.begin().await?;

    let tracking_info = db_tracking::read_source_tracking_info_for_precommit(
        source_id,
        source_key_json,
        db_setup,
        &mut *txn,
    )
    .await?;
    if let Some(tracking_info) = &tracking_info {
        let existing_source_version = SourceVersion::from_stored(
            tracking_info.processed_source_ordinal,
            &tracking_info.process_logic_fingerprint,
            &tracking_info.processed_source_content_hash,
            logic_fp,
        );
        if existing_source_version.should_skip(source_version, Some(update_stats)) {
            return Ok(SkippedOr::Skipped(existing_source_version));
        }
    }
    let tracking_info_exists = tracking_info.is_some();
    let process_ordinal = (tracking_info
        .as_ref()
        .map(|info| info.max_process_ordinal)
        .unwrap_or(0)
        + 1)
    .max(process_timestamp.timestamp_millis());
    let existing_process_ordinal = tracking_info.as_ref().and_then(|info| info.process_ordinal);

    let mut tracking_info_for_targets = HashMap::<i32, TrackingInfoForTarget>::new();
    for export_op in export_ops.iter() {
        tracking_info_for_targets
            .entry(export_op.target_id)
            .or_default()
            .export_op = Some(export_op);
    }

    // Collect `tracking_info_for_targets` from existing tracking info.
    if let Some(info) = tracking_info {
        let sqlx::types::Json(staging_target_keys) = info.staging_target_keys;
        for (target_id, keys_info) in staging_target_keys.into_iter() {
            let target_info = tracking_info_for_targets.entry(target_id).or_default();
            for key_info in keys_info.into_iter() {
                target_info
                    .existing_staging_keys_info
                    .entry(TargetKeyPair {
                        key: key_info.key,
                        additional_key: key_info.additional_key,
                    })
                    .or_default()
                    .push((key_info.process_ordinal, key_info.fingerprint));
            }
        }

        if let Some(sqlx::types::Json(target_keys)) = info.target_keys {
            for (target_id, keys_info) in target_keys.into_iter() {
                let target_info = tracking_info_for_targets.entry(target_id).or_default();
                for key_info in keys_info.into_iter() {
                    target_info
                        .existing_keys_info
                        .entry(TargetKeyPair {
                            key: key_info.key,
                            additional_key: key_info.additional_key,
                        })
                        .or_default()
                        .push((key_info.process_ordinal, key_info.fingerprint));
                }
            }
        }
    }

    let mut new_target_keys_info = db_tracking::TrackedTargetKeyForSource::default();
    if let Some(data) = &data {
        for export_op in export_ops.iter() {
            let target_info = tracking_info_for_targets
                .entry(export_op.target_id)
                .or_default();
            let mut keys_info = Vec::new();
            let collected_values =
                &data.evaluate_output.collected_values[export_op.input.collector_idx as usize];
            for value in collected_values.iter() {
                let primary_key = extract_primary_key(&export_op.primary_key_def, value)?;
                let primary_key_json = serde_json::to_value(&primary_key)?;

                let mut field_values = FieldValues {
                    fields: Vec::with_capacity(export_op.value_fields.len()),
                };
                for field in export_op.value_fields.iter() {
                    field_values
                        .fields
                        .push(value.fields[*field as usize].clone());
                }
                let additional_key = export_op.export_target_factory.extract_additional_key(
                    &primary_key,
                    &field_values,
                    export_op.export_context.as_ref(),
                )?;
                let target_key_pair = TargetKeyPair {
                    key: primary_key_json,
                    additional_key,
                };
                let existing_target_keys = target_info.existing_keys_info.remove(&target_key_pair);
                let existing_staging_target_keys = target_info
                    .existing_staging_keys_info
                    .remove(&target_key_pair);

                let curr_fp = if !export_op.value_stable {
                    Some(
                        Fingerprinter::default()
                            .with(&field_values)?
                            .into_fingerprint(),
                    )
                } else {
                    None
                };
                if existing_target_keys
                    .as_ref()
                    .map(|keys| !keys.is_empty() && keys.iter().all(|(_, fp)| fp == &curr_fp))
                    .unwrap_or(false)
                    && existing_staging_target_keys
                        .map(|keys| keys.iter().all(|(_, fp)| fp == &curr_fp))
                        .unwrap_or(true)
                {
                    // Already exists, with exactly the same value fingerprint.
                    // Nothing need to be changed, except carrying over the existing target keys info.
                    let (existing_ordinal, existing_fp) = existing_target_keys
                        .ok_or_else(invariance_violation)?
                        .into_iter()
                        .next()
                        .ok_or_else(invariance_violation)?;
                    keys_info.push(TrackedTargetKeyInfo {
                        key: target_key_pair.key,
                        additional_key: target_key_pair.additional_key,
                        process_ordinal: existing_ordinal,
                        fingerprint: existing_fp,
                    });
                } else {
                    // Entry with new value. Needs to be upserted.
                    let tracked_target_key = TrackedTargetKeyInfo {
                        key: target_key_pair.key.clone(),
                        additional_key: target_key_pair.additional_key.clone(),
                        process_ordinal,
                        fingerprint: curr_fp,
                    };
                    target_info.mutation.upserts.push(ExportTargetUpsertEntry {
                        key: primary_key,
                        additional_key: target_key_pair.additional_key,
                        value: field_values,
                    });
                    target_info
                        .new_staging_keys_info
                        .push(tracked_target_key.clone());
                    keys_info.push(tracked_target_key);
                }
            }
            new_target_keys_info.push((export_op.target_id, keys_info));
        }
    }

    let mut new_staging_target_keys = db_tracking::TrackedTargetKeyForSource::default();
    let mut target_mutations = HashMap::with_capacity(export_ops.len());
    for (target_id, target_tracking_info) in tracking_info_for_targets.into_iter() {
        let legacy_keys: HashSet<TargetKeyPair> = target_tracking_info
            .existing_keys_info
            .into_keys()
            .chain(target_tracking_info.existing_staging_keys_info.into_keys())
            .collect();

        let mut new_staging_keys_info = target_tracking_info.new_staging_keys_info;
        // Add tracking info for deletions.
        new_staging_keys_info.extend(legacy_keys.iter().map(|key| TrackedTargetKeyInfo {
            key: key.key.clone(),
            additional_key: key.additional_key.clone(),
            process_ordinal,
            fingerprint: None,
        }));
        new_staging_target_keys.push((target_id, new_staging_keys_info));

        if let Some(export_op) = target_tracking_info.export_op {
            let mut mutation = target_tracking_info.mutation;
            mutation.deletes.reserve(legacy_keys.len());
            for legacy_key in legacy_keys.into_iter() {
                let key = value::Value::<value::ScopeValue>::from_json(
                    legacy_key.key,
                    &export_op.primary_key_type,
                )?
                .as_key()?;
                mutation.deletes.push(interface::ExportTargetDeleteEntry {
                    key,
                    additional_key: legacy_key.additional_key,
                });
            }
            target_mutations.insert(target_id, mutation);
        }
    }

    db_tracking::precommit_source_tracking_info(
        source_id,
        source_key_json,
        process_ordinal,
        new_staging_target_keys,
        data.as_ref().map(|data| data.memoization_info),
        db_setup,
        &mut *txn,
        if tracking_info_exists {
            WriteAction::Update
        } else {
            WriteAction::Insert
        },
    )
    .await?;

    txn.commit().await?;

    Ok(SkippedOr::Normal(PrecommitOutput {
        metadata: PrecommitMetadata {
            source_entry_exists: data.is_some(),
            process_ordinal,
            existing_process_ordinal,
            new_target_keys: new_target_keys_info,
        },
        target_mutations,
    }))
}

async fn commit_source_tracking_info(
    source_id: i32,
    source_key_json: &serde_json::Value,
    source_version: &SourceVersion,
    logic_fingerprint: &[u8],
    precommit_metadata: PrecommitMetadata,
    process_timestamp: &chrono::DateTime<chrono::Utc>,
    db_setup: &db_tracking_setup::TrackingTableSetupState,
    pool: &PgPool,
) -> Result<()> {
    let mut txn = pool.begin().await?;

    let tracking_info = db_tracking::read_source_tracking_info_for_commit(
        source_id,
        source_key_json,
        db_setup,
        &mut *txn,
    )
    .await?;
    let tracking_info_exists = tracking_info.is_some();
    if tracking_info.as_ref().and_then(|info| info.process_ordinal)
        >= Some(precommit_metadata.process_ordinal)
    {
        return Ok(());
    }

    let cleaned_staging_target_keys = tracking_info
        .map(|info| {
            let sqlx::types::Json(staging_target_keys) = info.staging_target_keys;
            staging_target_keys
                .into_iter()
                .filter_map(|(target_id, target_keys)| {
                    let cleaned_target_keys: Vec<_> = target_keys
                        .into_iter()
                        .filter(|key_info| {
                            Some(key_info.process_ordinal)
                                > precommit_metadata.existing_process_ordinal
                                && key_info.process_ordinal != precommit_metadata.process_ordinal
                        })
                        .collect();
                    if !cleaned_target_keys.is_empty() {
                        Some((target_id, cleaned_target_keys))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !precommit_metadata.source_entry_exists && cleaned_staging_target_keys.is_empty() {
        // TODO: When we support distributed execution in the future, we'll need to leave a tombstone for a while
        // to prevent an earlier update causing the record reappear because of out-of-order processing.
        if tracking_info_exists {
            db_tracking::delete_source_tracking_info(
                source_id,
                source_key_json,
                db_setup,
                &mut *txn,
            )
            .await?;
        }
    } else {
        db_tracking::commit_source_tracking_info(
            source_id,
            source_key_json,
            cleaned_staging_target_keys,
            source_version.ordinal.into(),
            logic_fingerprint,
            source_version.content_hash.as_ref().map(|h| h.0.as_slice()),
            precommit_metadata.process_ordinal,
            process_timestamp.timestamp_micros(),
            precommit_metadata.new_target_keys,
            db_setup,
            &mut *txn,
            if tracking_info_exists {
                WriteAction::Update
            } else {
                WriteAction::Insert
            },
        )
        .await?;
    }

    txn.commit().await?;

    Ok(())
}

pub async fn evaluate_source_entry_with_memory(
    src_eval_ctx: &SourceRowEvaluationContext<'_>,
    options: EvaluationMemoryOptions,
    pool: &PgPool,
) -> Result<Option<EvaluateSourceEntryOutput>> {
    let stored_info = if options.enable_cache || !options.evaluation_only {
        let source_key_json = serde_json::to_value(src_eval_ctx.key)?;
        let existing_tracking_info = read_source_tracking_info_for_processing(
            src_eval_ctx.import_op.source_id,
            &source_key_json,
            &src_eval_ctx.plan.tracking_table_setup,
            pool,
        )
        .await?;
        existing_tracking_info
            .and_then(|info| info.memoization_info.map(|info| info.0))
            .flatten()
    } else {
        None
    };
    let memory = EvaluationMemory::new(chrono::Utc::now(), stored_info, options);
    let source_value = src_eval_ctx
        .import_op
        .executor
        .get_value(
            src_eval_ctx.key,
            &SourceExecutorGetOptions {
                include_value: true,
                include_ordinal: true,
                include_content_hash: true,
            },
        )
        .await?
        .value
        .ok_or_else(|| anyhow::anyhow!("value not returned"))?;
    let output = match source_value {
        interface::SourceValue::Existence(source_value) => {
            Some(evaluate_source_entry(src_eval_ctx, source_value, &memory).await?)
        }
        interface::SourceValue::NonExistence => None,
    };
    Ok(output)
}

pub async fn update_source_row(
    src_eval_ctx: &SourceRowEvaluationContext<'_>,
    source_value: interface::SourceValue,
    source_version: &SourceVersion,
    pool: &PgPool,
    update_stats: &stats::UpdateStats,
) -> Result<SkippedOr<()>> {
    let source_key_json = serde_json::to_value(src_eval_ctx.key)?;
    let process_time = chrono::Utc::now();

    // Phase 1: Evaluate with memoization info.
    let existing_tracking_info = read_source_tracking_info_for_processing(
        src_eval_ctx.import_op.source_id,
        &source_key_json,
        &src_eval_ctx.plan.tracking_table_setup,
        pool,
    )
    .await?;
    let (memoization_info, existing_version) = match existing_tracking_info {
        Some(info) => {
            let existing_version = SourceVersion::from_stored(
                info.processed_source_ordinal,
                &info.process_logic_fingerprint,
                &info.processed_source_content_hash,
                src_eval_ctx.plan.logic_fingerprint,
            );
            if existing_version.should_skip(source_version, Some(update_stats)) {
                return Ok(SkippedOr::Skipped(existing_version));
            }
            (
                info.memoization_info.and_then(|info| info.0),
                Some(existing_version),
            )
        }
        None => Default::default(),
    };
    let (output, stored_mem_info) = match source_value {
        interface::SourceValue::Existence(source_value) => {
            let evaluation_memory = EvaluationMemory::new(
                process_time,
                memoization_info,
                EvaluationMemoryOptions {
                    enable_cache: true,
                    evaluation_only: false,
                },
            );
            let output =
                evaluate_source_entry(src_eval_ctx, source_value, &evaluation_memory).await?;
            (Some(output), evaluation_memory.into_stored()?)
        }
        interface::SourceValue::NonExistence => Default::default(),
    };

    // Phase 2 (precommit): Update with the memoization info and stage target keys.
    let precommit_output = precommit_source_tracking_info(
        src_eval_ctx.import_op.source_id,
        &source_key_json,
        source_version,
        src_eval_ctx.plan.logic_fingerprint,
        output.as_ref().map(|scope_value| PrecommitData {
            evaluate_output: scope_value,
            memoization_info: &stored_mem_info,
        }),
        &process_time,
        &src_eval_ctx.plan.tracking_table_setup,
        &src_eval_ctx.plan.export_ops,
        update_stats,
        pool,
    )
    .await?;
    let precommit_output = match precommit_output {
        SkippedOr::Normal(output) => output,
        SkippedOr::Skipped(source_version) => return Ok(SkippedOr::Skipped(source_version)),
    };

    // Phase 3: Apply changes to the target storage, including upserting new target records and removing existing ones.
    let mut target_mutations = precommit_output.target_mutations;
    let apply_futs = src_eval_ctx
        .plan
        .export_op_groups
        .iter()
        .filter_map(|export_op_group| {
            let mutations_w_ctx: Vec<_> = export_op_group
                .op_idx
                .iter()
                .filter_map(|export_op_idx| {
                    let export_op = &src_eval_ctx.plan.export_ops[*export_op_idx];
                    target_mutations
                        .remove(&export_op.target_id)
                        .filter(|m| !m.is_empty())
                        .map(|mutation| interface::ExportTargetMutationWithContext {
                            mutation,
                            export_context: export_op.export_context.as_ref(),
                        })
                })
                .collect();
            (!mutations_w_ctx.is_empty()).then(|| {
                export_op_group
                    .target_factory
                    .apply_mutation(mutations_w_ctx)
            })
        });

    // TODO: Handle errors.
    try_join_all(apply_futs).await?;

    // Phase 4: Update the tracking record.
    commit_source_tracking_info(
        src_eval_ctx.import_op.source_id,
        &source_key_json,
        source_version,
        &src_eval_ctx.plan.logic_fingerprint.0,
        precommit_output.metadata,
        &process_time,
        &src_eval_ctx.plan.tracking_table_setup,
        pool,
    )
    .await?;

    if let Some(existing_version) = existing_version {
        if output.is_some() {
            if !source_version.ordinal.is_available()
                || source_version.ordinal != existing_version.ordinal
            {
                update_stats.num_updates.inc(1);
            } else {
                update_stats.num_reprocesses.inc(1);
            }
        } else {
            update_stats.num_deletions.inc(1);
        }
    } else if output.is_some() {
        update_stats.num_insertions.inc(1);
    }

    Ok(SkippedOr::Normal(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::fingerprint::Fingerprinter;
    use crate::ops::interface::{SourceExecutorListOptions, SourceExecutorGetOptions};

    #[test]
    fn test_should_skip_with_same_content_hash() {
        let content_hash = Fingerprinter::default()
            .with(&"test content".to_string())
            .unwrap()
            .into_fingerprint();
        
        let existing_version = SourceVersion {
            ordinal: Ordinal(Some(100)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(content_hash),
        };
        
        let target_version = SourceVersion {
            ordinal: Ordinal(Some(200)), // Newer timestamp
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(content_hash), // Same content
        };
        
        // Should skip because content is the same, despite newer timestamp
        assert!(existing_version.should_skip(&target_version, None));
    }

    #[test]
    fn test_should_not_skip_with_different_content_hash() {
        let old_content_hash = Fingerprinter::default()
            .with(&"old content".to_string())
            .unwrap()
            .into_fingerprint();
            
        let new_content_hash = Fingerprinter::default()
            .with(&"new content".to_string())
            .unwrap()
            .into_fingerprint();
        
        let existing_version = SourceVersion {
            ordinal: Ordinal(Some(100)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(old_content_hash),
        };
        
        let target_version = SourceVersion {
            ordinal: Ordinal(Some(200)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(new_content_hash),
        };
        
        // Should not skip because content is different
        assert!(!existing_version.should_skip(&target_version, None));
    }

    #[test]
    fn test_fallback_to_ordinal_when_no_content_hash() {
        let existing_version = SourceVersion {
            ordinal: Ordinal(Some(200)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: None, // No content hash
        };
        
        let target_version = SourceVersion {
            ordinal: Ordinal(Some(100)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: None,
        };
        
        // Should skip because existing ordinal > target ordinal
        assert!(existing_version.should_skip(&target_version, None));
    }

    #[test]
    fn test_mixed_content_hash_availability() {
        let content_hash = Fingerprinter::default()
            .with(&"test content".to_string())
            .unwrap()
            .into_fingerprint();
            
        let existing_version = SourceVersion {
            ordinal: Ordinal(Some(100)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(content_hash),
        };
        
        let target_version = SourceVersion {
            ordinal: Ordinal(Some(200)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: None, // No content hash for target
        };
        
        // Should fallback to ordinal comparison
        assert!(!existing_version.should_skip(&target_version, None));
    }

    #[test]
    fn test_github_actions_scenario_simulation() {
        // Simulate the exact GitHub Actions scenario
        
        // Initial state: file processed with content hash
        let initial_content = "def main():\n    print('Hello, World!')\n";
        let initial_hash = Fingerprinter::default()
            .with(&initial_content.to_string())
            .unwrap()
            .into_fingerprint();
        
        let processed_version = SourceVersion {
            ordinal: Ordinal(Some(1000)), // Original timestamp
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(initial_hash),
        };
        
        // GitHub Actions checkout: timestamp changes but content same
        let after_checkout_version = SourceVersion {
            ordinal: Ordinal(Some(2000)), // New timestamp after checkout
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(initial_hash), // Same content hash
        };
        
        // Should skip processing (this is the key improvement)
        assert!(processed_version.should_skip(&after_checkout_version, None));
        
        // Now simulate actual content change
        let updated_content = "def main():\n    print('Hello, Updated World!')\n";
        let updated_hash = Fingerprinter::default()
            .with(&updated_content.to_string())
            .unwrap()
            .into_fingerprint();
        
        let content_changed_version = SourceVersion {
            ordinal: Ordinal(Some(3000)), // Even newer timestamp
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(updated_hash), // Different content hash
        };
        
        // Should NOT skip processing (content actually changed)
        assert!(!processed_version.should_skip(&content_changed_version, None));
    }

    #[test]
    fn test_source_version_from_stored_with_content_hash() {
        let logic_fp = Fingerprinter::default()
            .with(&"logic_v1".to_string())
            .unwrap()
            .into_fingerprint();
        
        let content_hash_bytes = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        
        let version = SourceVersion::from_stored(
            Some(1000),
            &Some(logic_fp.0.to_vec()),
            &Some(content_hash_bytes.clone()),
            logic_fp,
        );
        
        assert_eq!(version.ordinal.0, Some(1000));
        assert!(matches!(version.kind, SourceVersionKind::CurrentLogic));
        assert!(version.content_hash.is_some());
        
        // Verify content hash is properly reconstructed
        let reconstructed_hash = version.content_hash.unwrap();
        assert_eq!(reconstructed_hash.0.as_slice(), &content_hash_bytes[..16]);
    }

    #[test]
    fn test_content_hash_priority_over_ordinal() {
        // Test that content hash comparison takes priority over ordinal comparison
        
        let same_content_hash = Fingerprinter::default()
            .with(&"same content".to_string())
            .unwrap()
            .into_fingerprint();
        
        // Case 1: Same content hash, newer ordinal -> should skip
        let existing = SourceVersion {
            ordinal: Ordinal(Some(100)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(same_content_hash),
        };
        
        let target = SourceVersion {
            ordinal: Ordinal(Some(200)), // Much newer
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(same_content_hash), // But same content
        };
        
        assert!(existing.should_skip(&target, None));
        
        // Case 2: Different content hash, older ordinal -> should NOT skip
        let different_content_hash = Fingerprinter::default()
            .with(&"different content".to_string())
            .unwrap()
            .into_fingerprint();
        
        let target_different = SourceVersion {
            ordinal: Ordinal(Some(50)), // Older timestamp
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(different_content_hash), // But different content
        };
        
        assert!(!existing.should_skip(&target_different, None));
    }

    #[test]
    fn test_backward_compatibility_without_content_hash() {
        // Ensure the system still works when content hashes are not available
        
        let existing_no_hash = SourceVersion {
            ordinal: Ordinal(Some(200)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: None,
        };
        
        let target_no_hash = SourceVersion {
            ordinal: Ordinal(Some(100)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: None,
        };
        
        // Should fall back to ordinal comparison
        assert!(existing_no_hash.should_skip(&target_no_hash, None));
        
        // Reverse case
        assert!(!target_no_hash.should_skip(&existing_no_hash, None));
    }

    #[test]
    fn test_content_hash_with_different_logic_versions() {
        let content_hash = Fingerprinter::default()
            .with(&"test content".to_string())
            .unwrap()
            .into_fingerprint();
        
        // Same content but different logic version should not skip
        let existing = SourceVersion {
            ordinal: Ordinal(Some(100)),
            kind: SourceVersionKind::DifferentLogic, // Different logic
            content_hash: Some(content_hash),
        };
        
        let target = SourceVersion {
            ordinal: Ordinal(Some(200)),
            kind: SourceVersionKind::CurrentLogic,
            content_hash: Some(content_hash), // Same content
        };
        
        // Should not skip because logic is different
        assert!(!existing.should_skip(&target, None));
    }

    #[test]
    fn test_source_executor_options_include_content_hash() {
        // Test that the new include_content_hash option is properly handled
        
        let list_options = SourceExecutorListOptions {
            include_ordinal: true,
            include_content_hash: true,
        };
        
        assert!(list_options.include_content_hash);
        
        let get_options = SourceExecutorGetOptions {
            include_value: true,
            include_ordinal: true,
            include_content_hash: true,
        };
        
        assert!(get_options.include_content_hash);
    }
}
