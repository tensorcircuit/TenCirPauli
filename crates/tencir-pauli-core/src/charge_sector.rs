//! Exact additive-charge sector dynamic programming and basis indexing.

use rustc_hash::FxHashMap;

use crate::PauliError;

/// A reusable suffix-count plan for one or more additive charge constraints.
pub struct ChargeSectorPlan {
    local_dimensions: Vec<usize>,
    contributions: Vec<Vec<Vec<i128>>>,
    target: Vec<i128>,
    suffix_counts: Vec<FxHashMap<Vec<i128>, u128>>,
    dimension: usize,
    estimated_bytes: u128,
}

impl ChargeSectorPlan {
    /// Return the number of selected basis states.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Return the best-effort plan-size estimate.
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    /// Return the lexicographic rank of one selected occupation.
    pub fn rank(&self, occupations: &[u64]) -> Result<u64, PauliError> {
        if occupations.len() != self.local_dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.local_dimensions.len(),
                actual: occupations.len(),
            });
        }
        let values: Vec<usize> = occupations
            .iter()
            .enumerate()
            .map(|(position, &value)| {
                let value = usize::try_from(value).map_err(|_| PauliError::InvalidIndex {
                    context: "occupation is outside the finite sector layout",
                })?;
                if value >= self.local_dimensions[position] {
                    return Err(PauliError::InvalidIndex {
                        context: "occupation is outside the finite sector layout",
                    });
                }
                Ok(value)
            })
            .collect::<Result<_, _>>()?;

        let mut remaining = self.target.clone();
        let mut candidate_remaining = vec![0_i128; self.target.len()];
        let mut rank = 0_u128;
        for (position, &value) in values.iter().enumerate() {
            for candidate in 0..value {
                subtract_contribution_into(
                    &remaining,
                    &self.contributions[position][candidate],
                    &mut candidate_remaining,
                )?;
                rank = rank
                    .checked_add(
                        self.suffix_counts[position + 1]
                            .get(&candidate_remaining)
                            .copied()
                            .unwrap_or(0),
                    )
                    .ok_or(PauliError::Overflow {
                        context: "computing charge-sector rank",
                    })?;
            }
            subtract_contribution_into(
                &remaining,
                &self.contributions[position][value],
                &mut candidate_remaining,
            )?;
            std::mem::swap(&mut remaining, &mut candidate_remaining);
        }
        if remaining.iter().any(|&value| value != 0) {
            return Err(PauliError::InvalidSector {
                context: "occupation does not satisfy every charge constraint",
            });
        }
        u64::try_from(rank).map_err(|_| PauliError::Overflow {
            context: "converting charge-sector rank",
        })
    }

    /// Return one selected occupation by lexicographic rank.
    pub fn unrank(&self, index: u64) -> Result<Vec<u64>, PauliError> {
        if u128::from(index) >= self.dimension as u128 {
            return Err(PauliError::InvalidIndex {
                context: "sector index is out of range",
            });
        }
        let mut values = vec![0_u64; self.local_dimensions.len()];
        let mut remaining = vec![0_i128; self.target.len()];
        let mut candidate_remaining = vec![0_i128; self.target.len()];
        self.unrank_into(index, &mut values, &mut remaining, &mut candidate_remaining)?;
        Ok(values)
    }

    fn unrank_into(
        &self,
        index: u64,
        values: &mut [u64],
        remaining: &mut [i128],
        candidate_remaining: &mut [i128],
    ) -> Result<(), PauliError> {
        if values.len() != self.local_dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.local_dimensions.len(),
                actual: values.len(),
            });
        }
        remaining.copy_from_slice(&self.target);
        let mut index = u128::from(index);
        for (position, &dimension) in self.local_dimensions.iter().enumerate() {
            let mut selected = None;
            for candidate in 0..dimension {
                subtract_contribution_into(
                    remaining,
                    &self.contributions[position][candidate],
                    candidate_remaining,
                )?;
                let count = self.suffix_counts[position + 1]
                    .get(&*candidate_remaining)
                    .copied()
                    .unwrap_or(0);
                if index < count {
                    selected = Some(candidate);
                    break;
                }
                index -= count;
            }
            let Some(candidate) = selected else {
                return Err(PauliError::InvalidSector {
                    context: "charge-sector rank/unrank plan is inconsistent",
                });
            };
            values[position] = u64::try_from(candidate).map_err(|_| PauliError::Overflow {
                context: "converting charge-sector occupation",
            })?;
            remaining.copy_from_slice(candidate_remaining);
        }
        Ok(())
    }

    /// Materialize the selected occupations in one flat row-major buffer.
    pub fn basis_states(&self, max_bytes: u128) -> Result<Vec<u64>, PauliError> {
        let axis_count = self.local_dimensions.len();
        let requested = (self.dimension as u128)
            .checked_mul(axis_count.max(1) as u128)
            .and_then(|value| value.checked_mul(8))
            .ok_or(PauliError::Overflow {
                context: "estimating charge-sector basis states",
            })?;
        if requested > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested,
                limit: max_bytes,
            });
        }
        let output_len = self
            .dimension
            .checked_mul(axis_count)
            .ok_or(PauliError::Overflow {
                context: "allocating charge-sector basis states",
            })?;
        let mut output = Vec::with_capacity(output_len);
        let mut values = vec![0_u64; axis_count];
        let mut remaining = vec![0_i128; self.target.len()];
        let mut candidate_remaining = vec![0_i128; self.target.len()];
        for index in 0..self.dimension {
            self.unrank_into(
                index as u64,
                &mut values,
                &mut remaining,
                &mut candidate_remaining,
            )?;
            output.extend_from_slice(&values);
        }
        Ok(output)
    }
}

/// Build a checked suffix-count dynamic-programming plan.
pub fn build_charge_sector_plan(
    local_dimensions: Vec<usize>,
    contributions: Vec<Vec<Vec<i128>>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> Result<ChargeSectorPlan, PauliError> {
    if contributions.len() != local_dimensions.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: local_dimensions.len(),
            actual: contributions.len(),
        });
    }
    if target.is_empty() {
        return Err(PauliError::InvalidSector {
            context: "charge-sector requires at least one charge constraint",
        });
    }
    for (position, (dimension, table)) in local_dimensions.iter().zip(&contributions).enumerate() {
        if table.len() != *dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: *dimension,
                actual: table.len(),
            });
        }
        if table
            .iter()
            .any(|contribution| contribution.len() != target.len())
        {
            return Err(PauliError::InvalidStructureLength {
                expected: target.len(),
                actual: table
                    .iter()
                    .find(|contribution| contribution.len() != target.len())
                    .map_or(0, Vec::len),
            });
        }
        if position == usize::MAX {
            return Err(PauliError::Overflow {
                context: "checking charge-sector dimensions",
            });
        }
    }

    let zero = vec![0_i128; target.len()];
    // Map iteration order is never observable: candidate traversal defines the
    // public lexicographic order, while lookup is the DP hot path.
    let mut suffix_counts = (0..=local_dimensions.len())
        .map(|_| FxHashMap::default())
        .collect::<Vec<FxHashMap<Vec<i128>, u128>>>();
    suffix_counts[local_dimensions.len()].insert(zero, 1);
    let entry_bytes = 64_u128
        .checked_add(
            24_u128
                .checked_mul(target.len() as u128)
                .ok_or(PauliError::Overflow {
                    context: "estimating charge-sector dynamic-programming plan",
                })?,
        )
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector dynamic-programming plan",
        })?;
    for position in (0..local_dimensions.len()).rev() {
        let mut table = FxHashMap::default();
        for contribution in &contributions[position] {
            for (remainder, &count) in &suffix_counts[position + 1] {
                let key = contribution
                    .iter()
                    .zip(remainder)
                    .map(|(&left, &right)| {
                        left.checked_add(right).ok_or(PauliError::Overflow {
                            context: "computing charge-sector contribution",
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let entry = table.entry(key).or_insert(0_u128);
                *entry = entry.checked_add(count).ok_or(PauliError::Overflow {
                    context: "counting charge-sector basis states",
                })?;
            }
        }
        check_plan_bytes(table.len(), entry_bytes, max_bytes)?;
        suffix_counts[position] = table;
    }

    let dimension_count = suffix_counts[0].get(&target).copied().unwrap_or(0);
    if dimension_count > isize::MAX as u128 {
        return Err(PauliError::Overflow {
            context: "converting charge-sector dimension to platform index",
        });
    }
    let dimension = usize::try_from(dimension_count).map_err(|_| PauliError::Overflow {
        context: "converting charge-sector dimension",
    })?;
    let table_bytes = suffix_counts.iter().try_fold(0_u128, |total, table| {
        let table_bytes =
            (table.len() as u128)
                .checked_mul(entry_bytes)
                .ok_or(PauliError::Overflow {
                    context: "estimating charge-sector plan",
                })?;
        total.checked_add(table_bytes).ok_or(PauliError::Overflow {
            context: "estimating charge-sector plan",
        })
    })?;
    let estimated_bytes = table_bytes
        .checked_add((local_dimensions.len() as u128).checked_mul(8).ok_or(
            PauliError::Overflow {
                context: "estimating charge-sector plan",
            },
        )?)
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(ChargeSectorPlan {
        local_dimensions,
        contributions,
        target,
        suffix_counts,
        dimension,
        estimated_bytes,
    })
}

fn subtract_contribution_into(
    remaining: &[i128],
    contribution: &[i128],
    output: &mut [i128],
) -> Result<(), PauliError> {
    if output.len() != remaining.len() || output.len() != contribution.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: remaining.len(),
            actual: output.len(),
        });
    }
    for ((slot, &left), &right) in output.iter_mut().zip(remaining).zip(contribution) {
        *slot = left.checked_sub(right).ok_or(PauliError::Overflow {
            context: "computing charge-sector remainder",
        })?;
    }
    Ok(())
}

fn check_plan_bytes(
    entries: usize,
    bytes_per_entry: u128,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let requested = (entries as u128)
        .checked_mul(bytes_per_entry)
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector dynamic-programming plan",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_unrank_and_basis_follow_mixed_radix_order() {
        let plan = build_charge_sector_plan(
            vec![2, 2, 2],
            vec![
                vec![vec![0], vec![1]],
                vec![vec![0], vec![1]],
                vec![vec![0], vec![1]],
            ],
            vec![1],
            u128::MAX,
        )
        .expect("charge sector plan");
        assert_eq!(plan.dimension(), 3);
        assert_eq!(plan.unrank(0).expect("unrank"), vec![0, 0, 1]);
        assert_eq!(plan.unrank(2).expect("unrank"), vec![1, 0, 0]);
        assert_eq!(plan.rank(&[0, 1, 0]).expect("rank"), 1);
        assert_eq!(
            plan.basis_states(u128::MAX).expect("basis"),
            vec![0, 0, 1, 0, 1, 0, 1, 0, 0]
        );
    }
}
