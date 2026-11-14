#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include "kmpc/types.hpp"
namespace kmpc {
// Sampling controls geometry accuracy, not the 20 ms execution clock. Consume
// every complete cell inside the joint-step box, then bisect only the first
// infeasible cell. Never skip an intervening path boundary.
template<class At>
double bounded_path_progress(double start,double end,const Vec18& origin,
                             const Vec18& cap,const At& at) {
  auto fits=[&](double s) {return ((at(s)-origin).cwiseAbs().array()<=cap.array()+1e-12).all();};
  if(!fits(start)) return start;
  double lo=start;
  while(lo<end-1e-12) {
    const double boundary=std::min(end,std::floor(lo+1e-9)+1.0);
    if(fits(boundary)) {lo=boundary;continue;}
    double hi=boundary;
    for(int it=0;it<28;++it) {
      const double mid=(lo+hi)*.5;
      if(fits(mid))lo=mid;else hi=mid;
    }
    break;
  }
  return lo;
}

// True iff every joint step from origin to q fits inside the velocity box.
inline bool within_velocity_cap(const Vec18& origin, const Vec18& q,
                                const Vec18& cap, double eps = 3e-7) {
  return ((q - origin).cwiseAbs().array() <= cap.array() + eps).all();
}

// Joint-space substeps needed to walk origin→target inside the per-tick cap.
// Returns 0 when already there, -1 when a joint has non-zero delta and non-positive cap.
inline int count_cap_substeps(const Vec18& origin, const Vec18& target,
                              const Vec18& cap, double eps = 3e-7) {
  int steps = 1;
  bool any = false;
  for (int j = 0; j < kNDof; ++j) {
    const double dq = std::abs(target[j] - origin[j]);
    if (dq <= eps) {
      continue;
    }
    any = true;
    if (!(cap[j] > eps)) {
      return -1;
    }
    steps = std::max(steps, static_cast<int>(std::ceil(dq / (cap[j] + eps))));
  }
  return any ? steps : 0;
}

// After a projected first stage, re-anchor onto a path parameter whose nominal
// sample lies inside the velocity box from `origin`. Prefer the closest such
// sample in [lo, hi]; returns nullopt when none exists (caller may bridge).
template <class At>
inline std::optional<double> find_progress_within_cap(
    double lo, double hi, const Vec18& origin, const Vec18& cap, const At& at,
    int samples = 64) {
  if (!(hi >= lo - 1e-15) || !origin.allFinite() || !cap.allFinite()) {
    return std::nullopt;
  }
  if (std::abs(hi - lo) <= 1e-12) {
    return within_velocity_cap(origin, at(lo), cap) ? std::optional<double>(lo)
                                                    : std::nullopt;
  }
  const int n = std::max(8, samples);
  double best_s = lo;
  double best = 1e300;
  bool found = false;
  for (int i = 0; i <= n; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(n);
    const double s = lo + u * (hi - lo);
    const Vec18 q = at(s);
    if (!within_velocity_cap(origin, q, cap)) {
      continue;
    }
    const double r = (q - origin).cwiseAbs().maxCoeff();
    if (r + 1e-15 < best) {
      best = r;
      best_s = s;
      found = true;
    }
  }
  if (!found) {
    return std::nullopt;
  }
  // Local refine around the coarse minimum.
  double a = std::max(lo, best_s - (hi - lo) / n);
  double b = std::min(hi, best_s + (hi - lo) / n);
  for (int it = 0; it < 24; ++it) {
    const double m1 = a + (b - a) / 3.0;
    const double m2 = b - (b - a) / 3.0;
    const Vec18 q1 = at(m1);
    const Vec18 q2 = at(m2);
    const bool ok1 = within_velocity_cap(origin, q1, cap);
    const bool ok2 = within_velocity_cap(origin, q2, cap);
    const double r1 = ok1 ? (q1 - origin).cwiseAbs().maxCoeff() : 1e300;
    const double r2 = ok2 ? (q2 - origin).cwiseAbs().maxCoeff() : 1e300;
    if (r1 < r2) {
      b = m2;
      if (ok1 && r1 < best) {
        best = r1;
        best_s = m1;
      }
    } else {
      a = m1;
      if (ok2 && r2 < best) {
        best = r2;
        best_s = m2;
      }
    }
  }
  return within_velocity_cap(origin, at(best_s), cap) ? std::optional<double>(best_s)
                                                      : std::nullopt;
}

// Map an accepted joint command back onto path parameter s∈[old,new].
// Never treat chord scale alone as an s fraction. Residual above threshold,
// ambiguous minima, or zero-span with large residual → unknown (do not guess).
struct ProgressRecoverResult {
  bool ok = false;
  bool unknown = false;
  double prog = 0.0;
  double residual_rad = 0.0;
  double second_best_residual_rad = 0.0;
  char reason[96]{"unset"};
};

template <class At>
inline ProgressRecoverResult recover_path_progress(
    double old_prog, double new_prog, const Vec18& accepted, const At& at,
    double max_residual_rad = 0.015, int samples = 64) {
  ProgressRecoverResult out;
  out.prog = old_prog;
  if (!accepted.allFinite() || !(new_prog >= old_prog - 1e-15)) {
    out.unknown = true;
    std::snprintf(out.reason, sizeof(out.reason), "invalid_interval_or_q");
    return out;
  }
  if (std::abs(new_prog - old_prog) <= 1e-12) {
    out.residual_rad = (at(old_prog) - accepted).cwiseAbs().maxCoeff();
    if (out.residual_rad > max_residual_rad) {
      out.unknown = true;
      std::snprintf(out.reason, sizeof(out.reason),
                    "zero_span_residual_%.4g", out.residual_rad);
      return out;
    }
    out.ok = true;
    out.prog = old_prog;
    std::snprintf(out.reason, sizeof(out.reason), "zero_span");
    return out;
  }
  const Vec18 at_new = at(new_prog);
  const double end_res = (at_new - accepted).cwiseAbs().maxCoeff();
  if (end_res <= 1e-9) {
    out.ok = true;
    out.prog = new_prog;
    out.residual_rad = end_res;
    std::snprintf(out.reason, sizeof(out.reason), "match_new");
    return out;
  }
  const int n = std::max(8, samples);
  double best_s = old_prog;
  double best = 1e300;
  double second = 1e300;
  double second_s = old_prog;
  for (int i = 0; i <= n; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(n);
    const double s = old_prog + u * (new_prog - old_prog);
    const double r = (at(s) - accepted).cwiseAbs().maxCoeff();
    if (r + 1e-15 < best) {
      second = best;
      second_s = best_s;
      best = r;
      best_s = s;
    } else if (r + 1e-15 < second) {
      second = r;
      second_s = s;
    }
  }
  double lo = std::max(old_prog, best_s - (new_prog - old_prog) / n);
  double hi = std::min(new_prog, best_s + (new_prog - old_prog) / n);
  for (int it = 0; it < 24; ++it) {
    const double m1 = lo + (hi - lo) / 3.0;
    const double m2 = hi - (hi - lo) / 3.0;
    const double r1 = (at(m1) - accepted).cwiseAbs().maxCoeff();
    const double r2 = (at(m2) - accepted).cwiseAbs().maxCoeff();
    if (r1 < r2) {
      hi = m2;
      if (r1 < best) {
        best = r1;
        best_s = m1;
      }
    } else {
      lo = m1;
      if (r2 < best) {
        best = r2;
        best_s = m2;
      }
    }
  }
  out.residual_rad = best;
  out.second_best_residual_rad = second;
  out.prog = best_s;
  if (!(best <= max_residual_rad)) {
    out.unknown = true;
    std::snprintf(out.reason, sizeof(out.reason),
                  "residual_%.4g>tol_%.4g", best, max_residual_rad);
    return out;
  }
  // Ambiguous only when a second near-equal minimum is separated in s.
  if (second <= best + 0.25 * max_residual_rad && second < 1e299 &&
      std::abs(second_s - best_s) > 0.05 * std::max(new_prog - old_prog, 1e-6)) {
    out.unknown = true;
    std::snprintf(out.reason, sizeof(out.reason), "ambiguous_minima");
    return out;
  }
  const double knot = std::floor(old_prog + 1e-9) + 1.0;
  if (old_prog + 1e-9 < knot && knot < new_prog - 1e-9 &&
      best > 0.5 * max_residual_rad) {
    out.unknown = true;
    std::snprintf(out.reason, sizeof(out.reason), "cross_knot_uncertain");
    return out;
  }
  out.ok = true;
  std::snprintf(out.reason, sizeof(out.reason), "recovered");
  return out;
}
}
