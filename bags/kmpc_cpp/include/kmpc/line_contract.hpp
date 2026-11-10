#pragma once
// Strict direction contract for single-arm line paths.
// Hard mask for this batch: waist J1/J2/J3 + pitch sum.
// Yaw and arm axes are reported but not hard-gated unless mask expands later.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <Eigen/StdVector>

#include "kmpc/tracker_math.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

inline constexpr double kDirectionNumericEpsRad = 1e-7;
inline constexpr double kDirectionReportDeadbandDeg = 0.01;
inline constexpr double kDirectionQualityTargetDeg = 0.02;
// Absolute fold hard gate under strict mode (waist axes + pitch sum).
inline constexpr double kDirectionStrictFoldMaxRad =
    kDirectionReportDeadbandDeg * kPi / 180.0;

enum class DirectionAxisId : int {
  WaistJ1 = 0,
  WaistJ2 = 1,
  WaistJ3 = 2,
  PitchSum = 3,
  Count = 4,
};

struct DirectionAxisMask {
  bool waist_j1 = true;
  bool waist_j2 = true;
  bool waist_j3 = true;
  bool pitch_sum = true;
};

struct DirectionAxisStats {
  int sign = 0;  // -1, 0, +1 from net displacement
  double fold_rad = 0.0;
  double backward_rad = 0.0;
  double peak_rad = 0.0;
  double net_rad = 0.0;
  int first_violation = -1;
};

struct DirectionCheckResult {
  bool ok = true;
  DirectionAxisMask mask{};
  DirectionAxisStats axes[static_cast<int>(DirectionAxisId::Count)]{};
  int violated_axis = -1;  // DirectionAxisId or -1
  char reason[64]{};
};

struct DirectionContract {
  uint64_t contract_id = 0;
  uint64_t chain_id = 0;
  uint64_t epoch = 0;
  DirectionAxisMask mask{};
  int sign[static_cast<int>(DirectionAxisId::Count)]{};
  Vec18 q_anchor = Vec18::Zero();
  double pitch_anchor = 0.0;
  double numeric_eps_rad = kDirectionNumericEpsRad;
  double fold_max_rad = kDirectionStrictFoldMaxRad;
  bool strict = true;
};

inline int direction_sign_from_net(double net, double eps) {
  if (net > eps) {
    return 1;
  }
  if (net < -eps) {
    return -1;
  }
  return 0;
}

template <class SampleAt>
inline void series_fold_backward(int n, SampleAt sample_at, double eps,
                                 DirectionAxisStats* out) {
  if (out == nullptr || n < 2) {
    if (out != nullptr) {
      *out = DirectionAxisStats{};
    }
    return;
  }
  DirectionAxisStats st;
  const double front = sample_at(0);
  const double back = sample_at(n - 1);
  st.net_rad = back - front;
  st.sign = direction_sign_from_net(st.net_rad, eps);
  st.fold_rad = chord_turnback(n, sample_at);

  if (st.sign == 0) {
    double lo = front;
    double hi = front;
    for (int i = 0; i < n; ++i) {
      const double v = sample_at(i);
      lo = std::min(lo, v);
      hi = std::max(hi, v);
      if (st.first_violation < 0 && std::abs(v - front) > eps) {
        // Near-zero net: any excursion beyond eps is a freeze violation site.
        st.first_violation = i;
      }
    }
    st.peak_rad = hi - lo;
    st.backward_rad = st.peak_rad;
    st.fold_rad = st.peak_rad;
  } else {
    const double dir = static_cast<double>(st.sign);
    double best = dir * front;
    double worst = 0.0;
    double back_sum = 0.0;
    for (int i = 0; i < n; ++i) {
      const double p = dir * sample_at(i);
      best = std::max(best, p);
      const double back_i = best - p;
      worst = std::max(worst, back_i);
      if (i > 0) {
        const double step = dir * (sample_at(i) - sample_at(i - 1));
        if (step < -eps) {
          back_sum += -step;
          if (st.first_violation < 0) {
            st.first_violation = i;
          }
        }
      }
    }
    st.peak_rad = worst;
    st.backward_rad = back_sum;
    st.fold_rad = std::max(st.fold_rad, worst);
  }
  *out = st;
}

inline DirectionContract make_path_direction_contract(
    const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& path,
    const DirectionAxisMask& mask = {},
    double eps = kDirectionNumericEpsRad) {
  DirectionContract c;
  c.mask = mask;
  c.numeric_eps_rad = eps;
  c.strict = true;
  if (path.empty()) {
    return c;
  }
  c.q_anchor = path.front();
  c.pitch_anchor = path.front().head<3>().sum();
  const Vec18& q0 = path.front();
  const Vec18& qN = path.back();
  for (int a = 0; a < 3; ++a) {
    c.sign[a] = direction_sign_from_net(qN[a] - q0[a], eps);
  }
  c.sign[static_cast<int>(DirectionAxisId::PitchSum)] =
      direction_sign_from_net(qN.head<3>().sum() - q0.head<3>().sum(), eps);
  return c;
}

inline DirectionCheckResult check_path_direction(
    const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& path,
    const DirectionContract& contract) {
  DirectionCheckResult r;
  r.mask = contract.mask;
  std::snprintf(r.reason, sizeof(r.reason), "ok");
  if (path.size() < 2) {
    return r;
  }
  const int n = static_cast<int>(path.size());
  const double eps = contract.numeric_eps_rad;

  auto fail = [&](DirectionAxisId id, const char* why) {
    r.ok = false;
    r.violated_axis = static_cast<int>(id);
    std::snprintf(r.reason, sizeof(r.reason), "%s", why);
  };

  const bool active[4] = {contract.mask.waist_j1, contract.mask.waist_j2,
                          contract.mask.waist_j3, contract.mask.pitch_sum};
  for (int a = 0; a < 3; ++a) {
    series_fold_backward(
        n, [&](int i) { return path[static_cast<std::size_t>(i)][a]; }, eps,
        &r.axes[a]);
    // Prefer contract sign when provided; otherwise use measured net.
    if (contract.sign[a] != 0) {
      r.axes[a].sign = contract.sign[a];
    }
    if (!active[a] || !contract.strict) {
      continue;
    }
    if (r.axes[a].fold_rad > contract.fold_max_rad + eps ||
        r.axes[a].backward_rad > contract.fold_max_rad + eps) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "direction_waist_j%d_fold", a + 1);
      fail(static_cast<DirectionAxisId>(a), buf);
      return r;
    }
    // Step-wise monotonicity under the committed sign.
    const int sgn = r.axes[a].sign;
    if (sgn == 0) {
      for (int i = 0; i < n; ++i) {
        if (std::abs(path[static_cast<std::size_t>(i)][a] - contract.q_anchor[a]) >
            contract.fold_max_rad + eps) {
          fail(static_cast<DirectionAxisId>(a), "direction_waist_frozen");
          return r;
        }
      }
    } else {
      for (int i = 1; i < n; ++i) {
        const double step =
            static_cast<double>(sgn) *
            (path[static_cast<std::size_t>(i)][a] -
             path[static_cast<std::size_t>(i - 1)][a]);
        if (step < -eps) {
          fail(static_cast<DirectionAxisId>(a), "direction_waist_step");
          return r;
        }
      }
    }
  }

  series_fold_backward(
      n,
      [&](int i) { return path[static_cast<std::size_t>(i)].head<3>().sum(); },
      eps, &r.axes[static_cast<int>(DirectionAxisId::PitchSum)]);
  if (contract.sign[static_cast<int>(DirectionAxisId::PitchSum)] != 0) {
    r.axes[static_cast<int>(DirectionAxisId::PitchSum)].sign =
        contract.sign[static_cast<int>(DirectionAxisId::PitchSum)];
  }
  if (active[3] && contract.strict) {
    const auto& ps = r.axes[static_cast<int>(DirectionAxisId::PitchSum)];
    if (ps.fold_rad > contract.fold_max_rad + eps ||
        ps.backward_rad > contract.fold_max_rad + eps) {
      fail(DirectionAxisId::PitchSum, "direction_pitch_fold");
      return r;
    }
    const int sgn = ps.sign;
    if (sgn == 0) {
      for (int i = 0; i < n; ++i) {
        const double p = path[static_cast<std::size_t>(i)].head<3>().sum();
        if (std::abs(p - contract.pitch_anchor) > contract.fold_max_rad + eps) {
          fail(DirectionAxisId::PitchSum, "direction_pitch_frozen");
          return r;
        }
      }
    } else {
      for (int i = 1; i < n; ++i) {
        const double step =
            static_cast<double>(sgn) *
            (path[static_cast<std::size_t>(i)].head<3>().sum() -
             path[static_cast<std::size_t>(i - 1)].head<3>().sum());
        if (step < -eps) {
          fail(DirectionAxisId::PitchSum, "direction_pitch_step");
          return r;
        }
      }
    }
  }
  return r;
}

inline DirectionCheckResult check_path_direction_strict(
    const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& path) {
  const DirectionContract c = make_path_direction_contract(path);
  return check_path_direction(path, c);
}

// Final-send / stream tick gate: one step against an evolving envelope.
struct DirectionEnvelope {
  bool active = false;
  DirectionAxisMask mask{};
  int sign[static_cast<int>(DirectionAxisId::Count)]{};
  double extremum[static_cast<int>(DirectionAxisId::Count)]{};
  Vec18 last_q = Vec18::Zero();
  double last_pitch = 0.0;
  double numeric_eps_rad = kDirectionNumericEpsRad;
  double fold_max_rad = kDirectionStrictFoldMaxRad;
};

inline void direction_envelope_reset(DirectionEnvelope* env) {
  if (env == nullptr) {
    return;
  }
  *env = DirectionEnvelope{};
}

inline void direction_envelope_seed(DirectionEnvelope* env,
                                    const DirectionContract& contract,
                                    const Vec18& q0) {
  if (env == nullptr) {
    return;
  }
  env->active = true;
  env->mask = contract.mask;
  env->numeric_eps_rad = contract.numeric_eps_rad;
  env->fold_max_rad = contract.fold_max_rad;
  for (int i = 0; i < static_cast<int>(DirectionAxisId::Count); ++i) {
    env->sign[i] = contract.sign[i];
  }
  env->last_q = q0;
  env->last_pitch = q0.head<3>().sum();
  env->extremum[0] = q0[0];
  env->extremum[1] = q0[1];
  env->extremum[2] = q0[2];
  env->extremum[3] = env->last_pitch;
}

inline DirectionCheckResult direction_envelope_accept(
    DirectionEnvelope* env, const Vec18& q_next) {
  DirectionCheckResult r;
  std::snprintf(r.reason, sizeof(r.reason), "ok");
  if (env == nullptr || !env->active) {
    return r;
  }
  r.mask = env->mask;
  const double eps = env->numeric_eps_rad;
  auto fail = [&](DirectionAxisId id, const char* why) {
    r.ok = false;
    r.violated_axis = static_cast<int>(id);
    std::snprintf(r.reason, sizeof(r.reason), "%s", why);
  };

  const bool active[4] = {env->mask.waist_j1, env->mask.waist_j2,
                          env->mask.waist_j3, env->mask.pitch_sum};
  for (int a = 0; a < 3; ++a) {
    if (!active[a]) {
      continue;
    }
    const int sgn = env->sign[a];
    const double step = q_next[a] - env->last_q[a];
    if (sgn == 0) {
      if (std::abs(q_next[a] - env->extremum[a]) > env->fold_max_rad + eps) {
        fail(static_cast<DirectionAxisId>(a), "final_send_waist_frozen");
        return r;
      }
    } else if (static_cast<double>(sgn) * step < -eps) {
      fail(static_cast<DirectionAxisId>(a), "final_send_waist_step");
      return r;
    } else {
      if (sgn > 0) {
        env->extremum[a] = std::max(env->extremum[a], q_next[a]);
      } else {
        env->extremum[a] = std::min(env->extremum[a], q_next[a]);
      }
      // Envelope: never go back past the extremum by more than eps.
      if (sgn > 0 && env->extremum[a] - q_next[a] > env->fold_max_rad + eps) {
        fail(static_cast<DirectionAxisId>(a), "final_send_waist_envelope");
        return r;
      }
      if (sgn < 0 && q_next[a] - env->extremum[a] > env->fold_max_rad + eps) {
        fail(static_cast<DirectionAxisId>(a), "final_send_waist_envelope");
        return r;
      }
    }
  }

  if (active[3]) {
    const double p = q_next.head<3>().sum();
    const int sgn = env->sign[3];
    const double step = p - env->last_pitch;
    if (sgn == 0) {
      if (std::abs(p - env->extremum[3]) > env->fold_max_rad + eps) {
        fail(DirectionAxisId::PitchSum, "final_send_pitch_frozen");
        return r;
      }
    } else if (static_cast<double>(sgn) * step < -eps) {
      fail(DirectionAxisId::PitchSum, "final_send_pitch_step");
      return r;
    } else {
      if (sgn > 0) {
        env->extremum[3] = std::max(env->extremum[3], p);
      } else {
        env->extremum[3] = std::min(env->extremum[3], p);
      }
      if (sgn > 0 && env->extremum[3] - p > env->fold_max_rad + eps) {
        fail(DirectionAxisId::PitchSum, "final_send_pitch_envelope");
        return r;
      }
      if (sgn < 0 && p - env->extremum[3] > env->fold_max_rad + eps) {
        fail(DirectionAxisId::PitchSum, "final_send_pitch_envelope");
        return r;
      }
    }
    env->last_pitch = p;
  }

  env->last_q = q_next;
  return r;
}

}  // namespace kmpc
