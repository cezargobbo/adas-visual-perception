#pragma once
// Linear assignment equal to lap.lapjv(cost, extend_cost=True, cost_limit=...) as used by
// ultralytics.trackers.utils.matching.linear_assignment.
#include <utility>
#include <vector>

namespace adasvp {

// Returns matched (row, col) pairs ordered by row. unmatched_a / unmatched_b (optional) receive the
// unassigned rows / columns in ascending order. Empty matrices match nothing; for a 0xN matrix (no rows to
// read N from) pass N as ncols_if_no_rows so every column is reported unmatched.
std::vector<std::pair<int, int>> linear_assignment(const std::vector<std::vector<double>>& cost,
                                                   double cost_limit, std::vector<int>* unmatched_a,
                                                   std::vector<int>* unmatched_b, int ncols_if_no_rows = 0);

}  // namespace adasvp
