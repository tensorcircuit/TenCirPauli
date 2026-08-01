use std::collections::BTreeSet;

use crate::error::PauliError;
use crate::word::PauliWord;

/// Compatibility relation used by deterministic measurement grouping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupingMode {
    /// Qubit-wise commuting: every local pair is equal or contains identity.
    QubitWise,
    /// Algebraic commuting: the binary symplectic inner product is zero.
    General,
}

/// Deterministic graph-coloring strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupingAlgorithm {
    /// Order vertices by descending incompatibility degree.
    LargestFirst,
    /// Repeatedly choose the highest saturation-degree vertex.
    Dsatur,
}

/// Default upper bound for the dense incompatibility matrix used by coloring.
pub const DEFAULT_MAX_GROUPING_ENTRIES: usize = 10_000_000;

/// Return a deterministic partition of input term indices into compatible groups.
pub fn group_words(
    words: &[PauliWord],
    mode: GroupingMode,
    algorithm: GroupingAlgorithm,
) -> Result<Vec<Vec<usize>>, PauliError> {
    group_words_bounded(words, mode, algorithm, DEFAULT_MAX_GROUPING_ENTRIES)
}

/// Return a deterministic partition with an explicit adjacency-entry limit.
pub fn group_words_bounded(
    words: &[PauliWord],
    mode: GroupingMode,
    algorithm: GroupingAlgorithm,
    max_entries: usize,
) -> Result<Vec<Vec<usize>>, PauliError> {
    if let Some(first) = words.first() {
        if let Some(other) = words.iter().find(|word| word.nqubits != first.nqubits) {
            return Err(PauliError::IncompatibleQubitCounts {
                left: first.nqubits,
                right: other.nqubits,
            });
        }
    }
    let size = words.len();
    let entries = size.checked_mul(size).ok_or(PauliError::Overflow {
        context: "estimating grouping matrix entries",
    })?;
    if entries > max_entries {
        return Err(PauliError::MemoryLimit {
            requested: entries as u128,
            limit: max_entries as u128,
        });
    }
    let mut incompatible = vec![vec![false; size]; size];
    for left in 0..size {
        for right in (left + 1)..size {
            let compatible = match mode {
                GroupingMode::QubitWise => words[left].qwc_compatible(&words[right])?,
                GroupingMode::General => words[left].commutes_with(&words[right])?,
            };
            incompatible[left][right] = !compatible;
            incompatible[right][left] = !compatible;
        }
    }
    let degrees = incompatible
        .iter()
        .map(|row| row.iter().filter(|value| **value).count())
        .collect::<Vec<_>>();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    if algorithm == GroupingAlgorithm::LargestFirst {
        let mut order: Vec<usize> = (0..size).collect();
        order.sort_by(|left, right| {
            degrees[*right]
                .cmp(&degrees[*left])
                .then_with(|| left.cmp(right))
        });
        for vertex in order {
            place_vertex(vertex, &incompatible, &mut groups);
        }
    } else {
        let mut colors = vec![None; size];
        let mut neighbor_colors = vec![BTreeSet::<usize>::new(); size];
        for _ in 0..size {
            let vertex = (0..size)
                .filter(|index| colors[*index].is_none())
                .max_by(|left, right| {
                    neighbor_colors[*left]
                        .len()
                        .cmp(&neighbor_colors[*right].len())
                        .then_with(|| degrees[*left].cmp(&degrees[*right]))
                        .then_with(|| right.cmp(left))
                })
                .expect("uncolored vertex exists for every DSATUR iteration");
            let color = (0..)
                .find(|color| !neighbor_colors[vertex].contains(color))
                .expect("a finite colored neighborhood always leaves an available color");
            colors[vertex] = Some(color);
            for other in 0..size {
                if colors[other].is_none() && incompatible[vertex][other] {
                    neighbor_colors[other].insert(color);
                }
            }
        }
        let max_color = colors.iter().flatten().copied().max().unwrap_or(0);
        groups = (0..=max_color).map(|_| Vec::new()).collect();
        for (vertex, color) in colors.into_iter().enumerate() {
            groups[color.expect("all DSATUR vertices are colored")].push(vertex);
        }
        groups.retain(|group| !group.is_empty());
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.sort_by_key(|group| group[0]);
    Ok(groups)
}

/// Build a bounded dense compatibility matrix for a batch of words.
pub fn compatibility_matrix(
    words: &[PauliWord],
    mode: GroupingMode,
    max_entries: usize,
) -> Result<Vec<bool>, PauliError> {
    let entries = words
        .len()
        .checked_mul(words.len())
        .ok_or(PauliError::Overflow {
            context: "estimating compatibility matrix entries",
        })?;
    if entries > max_entries {
        return Err(PauliError::MemoryLimit {
            requested: entries as u128,
            limit: max_entries as u128,
        });
    }
    let mut matrix = vec![false; entries];
    for left in 0..words.len() {
        for right in 0..words.len() {
            matrix[left * words.len() + right] = match mode {
                GroupingMode::QubitWise => words[left].qwc_compatible(&words[right])?,
                GroupingMode::General => words[left].commutes_with(&words[right])?,
            };
        }
    }
    Ok(matrix)
}

/// Build a bounded streaming incompatibility edge list without materializing
/// the dense matrix.
pub fn incompatibility_edges(
    words: &[PauliWord],
    mode: GroupingMode,
    max_edges: usize,
) -> Result<Vec<(usize, usize)>, PauliError> {
    let mut edges = Vec::new();
    for left in 0..words.len() {
        for right in (left + 1)..words.len() {
            let compatible = match mode {
                GroupingMode::QubitWise => words[left].qwc_compatible(&words[right])?,
                GroupingMode::General => words[left].commutes_with(&words[right])?,
            };
            if !compatible {
                if edges.len() >= max_edges {
                    return Err(PauliError::MemoryLimit {
                        requested: (edges.len() + 1) as u128,
                        limit: max_edges as u128,
                    });
                }
                edges.push((left, right));
            }
        }
    }
    Ok(edges)
}

fn place_vertex(vertex: usize, incompatible: &[Vec<bool>], groups: &mut Vec<Vec<usize>>) {
    if let Some(group) = groups
        .iter_mut()
        .find(|group| group.iter().all(|other| !incompatible[vertex][*other]))
    {
        group.push(vertex);
    } else {
        groups.push(vec![vertex]);
    }
}
