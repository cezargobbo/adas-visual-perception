#include "track/assignment.hpp"

#include <cstddef>
#include <limits>

namespace adasvp {
namespace {

// Dense Jonker-Volgenant (LAPJV) as in the lap package (column reduction, reduction transfer,
// two augmenting row reduction passes, augmentation). Following the same steps keeps the choice among
// equally optimal assignments identical to lap.
constexpr double kLarge = 1000000.0;  // lap's LARGE sentinel

class Lapjv {
 public:
  Lapjv(int n, const std::vector<double>& c) : n_(n), c_(c), x_(n, -1), y_(n, -1), v_(n, 0.0) {}

  void solve() {
    std::vector<int> free_rows(n_);
    int nfree = ccrrt(free_rows);
    for (int it = 0; nfree > 0 && it < 2; ++it) nfree = carr(nfree, free_rows);
    if (nfree > 0) ca(nfree, free_rows);
  }
  const std::vector<int>& x() const { return x_; }
  const std::vector<int>& y() const { return y_; }

 private:
  double C(int i, int j) const { return c_[static_cast<size_t>(i) * n_ + j]; }

  int ccrrt(std::vector<int>& free_rows) {
    for (int j = 0; j < n_; ++j) {
      v_[j] = kLarge;
      y_[j] = 0;
    }
    for (int i = 0; i < n_; ++i)
      for (int j = 0; j < n_; ++j) {
        const double c = C(i, j);
        if (c < v_[j]) {
          v_[j] = c;
          y_[j] = i;
        }
      }
    std::vector<char> unique(n_, 1);
    for (int j = n_ - 1; j >= 0; --j) {
      const int i = y_[j];
      if (x_[i] < 0) {
        x_[i] = j;
      } else {
        unique[i] = 0;
        y_[j] = -1;
      }
    }
    int nfree = 0;
    for (int i = 0; i < n_; ++i) {
      if (x_[i] < 0) {
        free_rows[nfree++] = i;
      } else if (unique[i]) {
        const int j = x_[i];
        double mn = kLarge;
        for (int j2 = 0; j2 < n_; ++j2) {
          if (j2 == j) continue;
          const double c = C(i, j2) - v_[j2];
          if (c < mn) mn = c;
        }
        v_[j] -= mn;
      }
    }
    return nfree;
  }

  int carr(int nfree, std::vector<int>& free_rows) {
    int current = 0, new_free = 0, rr_cnt = 0;
    while (current < nfree) {
      ++rr_cnt;
      const int free_i = free_rows[current++];
      int j1 = 0, j2 = -1;
      double v1 = C(free_i, 0) - v_[0], v2 = kLarge;
      for (int j = 1; j < n_; ++j) {
        const double c = C(free_i, j) - v_[j];
        if (c < v2) {
          if (c >= v1) {
            v2 = c;
            j2 = j;
          } else {
            v2 = v1;
            v1 = c;
            j2 = j1;
            j1 = j;
          }
        }
      }
      int i0 = y_[j1];
      const double v1_new = v_[j1] - (v2 - v1);
      const bool lowers = v1_new < v_[j1];
      if (rr_cnt < current * n_) {
        if (lowers) {
          v_[j1] = v1_new;
        } else if (i0 >= 0 && j2 >= 0) {
          j1 = j2;
          i0 = y_[j2];
        }
        if (i0 >= 0) {
          if (lowers)
            free_rows[--current] = i0;
          else
            free_rows[new_free++] = i0;
        }
      } else if (i0 >= 0) {
        free_rows[new_free++] = i0;
      }
      x_[free_i] = j1;
      y_[j1] = free_i;
    }
    return new_free;
  }

  int find(int lo, std::vector<double>& d, std::vector<int>& cols) {
    int hi = lo + 1;
    double mind = d[cols[lo]];
    for (int k = hi; k < n_; ++k) {
      const int j = cols[k];
      if (d[j] <= mind) {
        if (d[j] < mind) {
          hi = lo;
          mind = d[j];
        }
        cols[k] = cols[hi];
        cols[hi++] = j;
      }
    }
    return hi;
  }

  int scan(int& plo, int& phi, std::vector<double>& d, std::vector<int>& cols, std::vector<int>& pred) {
    int lo = plo, hi = phi;
    while (lo != hi) {
      int j = cols[lo++];
      const int i = y_[j];
      const double mind = d[j];
      const double h = C(i, j) - v_[j] - mind;
      for (int k = hi; k < n_; ++k) {
        j = cols[k];
        const double cred = C(i, j) - v_[j] - h;
        if (cred < d[j]) {
          d[j] = cred;
          pred[j] = i;
          if (cred == mind) {
            if (y_[j] < 0) return j;
            cols[k] = cols[hi];
            cols[hi++] = j;
          }
        }
      }
    }
    plo = lo;
    phi = hi;
    return -1;
  }

  int find_path(int start_i, std::vector<int>& pred) {
    std::vector<int> cols(n_);
    std::vector<double> d(n_);
    int lo = 0, hi = 0, final_j = -1, n_ready = 0;
    for (int i = 0; i < n_; ++i) {
      cols[i] = i;
      pred[i] = start_i;
      d[i] = C(start_i, i) - v_[i];
    }
    while (final_j == -1) {
      if (lo == hi) {
        n_ready = lo;
        hi = find(lo, d, cols);
        for (int k = lo; k < hi; ++k)
          if (y_[cols[k]] < 0) final_j = cols[k];
      }
      if (final_j == -1) final_j = scan(lo, hi, d, cols, pred);
    }
    const double mind = d[cols[lo]];
    for (int k = 0; k < n_ready; ++k) {
      const int j = cols[k];
      v_[j] += d[j] - mind;
    }
    return final_j;
  }

  void ca(int nfree, const std::vector<int>& free_rows) {
    std::vector<int> pred(n_);
    for (int f = 0; f < nfree; ++f) {
      const int free_i = free_rows[f];
      int i = -1;
      int j = find_path(free_i, pred);
      while (i != free_i) {
        i = pred[j];
        y_[j] = i;
        const int t = x_[i];
        x_[i] = j;
        j = t;
      }
    }
  }

  int n_;
  const std::vector<double>& c_;
  std::vector<int> x_, y_;
  std::vector<double> v_;
};

}  // namespace

std::vector<std::pair<int, int>> linear_assignment(const std::vector<std::vector<double>>& cost,
                                                   double cost_limit, std::vector<int>* unmatched_a,
                                                   std::vector<int>* unmatched_b, int ncols_if_no_rows) {
  const int nr = static_cast<int>(cost.size());
  const int nc = nr ? static_cast<int>(cost[0].size()) : ncols_if_no_rows;
  if (unmatched_a) unmatched_a->clear();
  if (unmatched_b) unmatched_b->clear();
  std::vector<std::pair<int, int>> matches;
  std::vector<int> x(nr, -1), y(nc, -1);
  if (nr > 0 && nc > 0) {
    // lap's extend_cost: (n+m)x(n+m), cost_limit/2 everywhere, zeros in the bottom-right block.
    const int n = nr + nc;
    std::vector<double> ext(static_cast<size_t>(n) * n, cost_limit / 2.0);
    for (int i = nr; i < n; ++i)
      for (int j = nc; j < n; ++j) ext[static_cast<size_t>(i) * n + j] = 0.0;
    for (int i = 0; i < nr; ++i)
      for (int j = 0; j < nc; ++j) ext[static_cast<size_t>(i) * n + j] = cost[i][j];
    Lapjv lap(n, ext);
    lap.solve();
    for (int i = 0; i < nr; ++i) x[i] = lap.x()[i] < nc ? lap.x()[i] : -1;
    for (int j = 0; j < nc; ++j) y[j] = lap.y()[j] < nr ? lap.y()[j] : -1;
  }
  for (int i = 0; i < nr; ++i) {
    if (x[i] >= 0)
      matches.emplace_back(i, x[i]);
    else if (unmatched_a)
      unmatched_a->push_back(i);
  }
  if (unmatched_b)
    for (int j = 0; j < nc; ++j)
      if (y[j] < 0) unmatched_b->push_back(j);
  return matches;
}

}  // namespace adasvp
