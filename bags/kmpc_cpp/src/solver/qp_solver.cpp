// ProxQP：36 维笛卡尔盒 + 最多 kMaxIneq=16 行；18 维关节空间抄同一热启动。
// kMaxIneq 编译期定维；C.resize 只在构造，每拍 topRows(n_ineq)。
// 清窗 x=0：不要带着上一题对偶。results.z 含盒对偶，不能塞进 solve(z)。

#include "kmpc/qp_solver.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <proxsuite/proxqp/dense/dense.hpp>
#pragma GCC diagnostic pop

namespace kmpc {
namespace {

constexpr double kIneqInf = 1.0e20;

template <typename MatA>
bool all_finite_top(const MatA& A, const VecIneq& b, int n_ineq) {
  for (int i = 0; i < n_ineq; ++i) {
    if (!A.row(i).allFinite() || !std::isfinite(b[i])) {
      return false;
    }
  }
  return true;
}

template <int Nx>
struct QpEngine {
  using Vec = Eigen::Matrix<double, Nx, 1>;
  using Mat = Eigen::Matrix<double, Nx, Nx>;
  using MatA = Eigen::Matrix<double, kMaxIneq, Nx>;
  using DenseQP = proxsuite::proxqp::dense::QP<double>;
  DenseQP qp;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C;
  Eigen::Matrix<double, Eigen::Dynamic, 1> g_dyn;
  Eigen::Matrix<double, Eigen::Dynamic, 1> l_in;
  Eigen::Matrix<double, Eigen::Dynamic, 1> u_in;
  Eigen::Matrix<double, Eigen::Dynamic, 1> l_box;
  Eigen::Matrix<double, Eigen::Dynamic, 1> u_box;
  bool initialized = false;

  QpEngine() : qp(Nx, 0, kMaxIneq, true) {
    H.resize(Nx, Nx);
    C.resize(kMaxIneq, Nx);
    g_dyn.resize(Nx);
    l_in.resize(kMaxIneq);
    u_in.resize(kMaxIneq);
    l_box.resize(Nx);
    u_box.resize(Nx);
    qp.settings.eps_abs = 1e-8;
    qp.settings.eps_rel = 1e-8;
    qp.settings.verbose = false;
    qp.settings.max_iter = 200;
  }
};

template <int Nx>
QpStatus solve_engine(
    QpEngine<Nx>& im,
    const Eigen::Matrix<double, Nx, Nx>& P,
    const Eigen::Matrix<double, Nx, 1>& g,
    const Eigen::Matrix<double, Nx, 1>& lb,
    const Eigen::Matrix<double, Nx, 1>& ub,
    const Eigen::Matrix<double, kMaxIneq, Nx>& A,
    const VecIneq& b,
    int n_ineq,
    Eigen::Matrix<double, Nx, 1>& x) {
  if (n_ineq < 0 || n_ineq > kMaxIneq) {
    x.setZero();
    return QpStatus::IllegalInput;
  }
  if (!P.allFinite() || !g.allFinite() || !lb.allFinite() || !ub.allFinite()) {
    x.setZero();
    return QpStatus::NonFinite;
  }
  if ((lb.array() > ub.array()).any()) {
    x.setZero();
    return QpStatus::IllegalInput;
  }
  if (!all_finite_top(A, b, n_ineq)) {
    x.setZero();
    return QpStatus::NonFinite;
  }

  im.H = P;
  im.g_dyn = g;
  im.l_box = lb;
  im.u_box = ub;
  im.C.setZero();
  im.l_in.setConstant(-kIneqInf);
  im.u_in.setConstant(kIneqInf);
  if (n_ineq > 0) {
    im.C.topRows(n_ineq) = A.topRows(n_ineq);
    im.l_in.head(n_ineq) = b.head(n_ineq);
    im.u_in.head(n_ineq).setConstant(kIneqInf);
  }

  try {
    using proxsuite::nullopt;
    if (!im.initialized) {
      im.qp.init(
          im.H, im.g_dyn, nullopt, nullopt, im.C, im.l_in, im.u_in, im.l_box,
          im.u_box, true);
      im.initialized = true;
    } else {
      im.qp.update(
          im.H, im.g_dyn, nullopt, nullopt, im.C, im.l_in, im.u_in, im.l_box,
          im.u_box, true);
    }
    if (x.cwiseAbs().maxCoeff() == 0.0) {
      im.qp.settings.initial_guess =
          proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
    } else {
      im.qp.results.x = x;
      im.qp.settings.initial_guess =
          proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    }
    im.qp.solve();
  } catch (const std::exception& ex) {
    x.setZero();
    std::fprintf(stderr, "kmpc QpSolver: %s\n", ex.what());
    return QpStatus::SolverFailed;
  } catch (...) {
    x.setZero();
    std::fprintf(stderr, "kmpc QpSolver: ProxQP exception\n");
    return QpStatus::SolverFailed;
  }

  const auto st = im.qp.results.info.status;
  if (st != proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED &&
      st != proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE) {
    x.setZero();
    std::fprintf(
        stderr, "kmpc QpSolver: ProxQP status=%d\n", static_cast<int>(st));
    return QpStatus::SolverFailed;
  }
  x = im.qp.results.x;
  if (!x.allFinite()) {
    x.setZero();
    std::fprintf(stderr, "kmpc QpSolver: non-finite x\n");
    return QpStatus::NonFinite;
  }
  x = x.cwiseMax(lb).cwiseMin(ub);
  return QpStatus::Success;
}

}  // namespace

struct QpSolver::Impl {
  QpEngine<kNx> e;
};

QpSolver::QpSolver() : impl_(std::make_unique<Impl>()) {}
QpSolver::QpSolver(QpSolver&&) noexcept = default;
QpSolver& QpSolver::operator=(QpSolver&&) noexcept = default;
QpSolver::~QpSolver() = default;

QpStatus QpSolver::solve_box_ineq(
    const Mat36& P,
    const Vec36& g,
    const Vec36& lb,
    const Vec36& ub,
    const MatIneq& A,
    const VecIneq& b,
    int n_ineq,
    Vec36& x) {
  return solve_engine(impl_->e, P, g, lb, ub, A, b, n_ineq, x);
}

struct QpSolver18::Impl {
  QpEngine<kNDof> e;
};

QpSolver18::QpSolver18() : impl_(std::make_unique<Impl>()) {}
QpSolver18::QpSolver18(QpSolver18&&) noexcept = default;
QpSolver18& QpSolver18::operator=(QpSolver18&&) noexcept = default;
QpSolver18::~QpSolver18() = default;

QpStatus QpSolver18::solve_box_ineq(
    const Mat18& P,
    const Vec18& g,
    const Vec18& lb,
    const Vec18& ub,
    const MatIneq18& A,
    const VecIneq& b,
    int n_ineq,
    Vec18& x) {
  return solve_engine(impl_->e, P, g, lb, ub, A, b, n_ineq, x);
}

}  // namespace kmpc
