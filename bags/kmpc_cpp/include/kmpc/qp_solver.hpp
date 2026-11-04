#pragma once
// 对应 Python kmpc/qp_solver.py 的盒+不等式接口。
// 生产求解器 ProxQP；热路径无 malloc。失败返回 QpStatus，不抛异常。
// QpSolver = 36 维笛卡尔；QpSolver18 = 关节空间。禁止把 36 维改成动态维。

#include <memory>

#include "kmpc/types.hpp"

namespace kmpc {

class QpSolver {
 public:
  QpSolver();
  QpSolver(const QpSolver&) = delete;
  QpSolver& operator=(const QpSolver&) = delete;
  QpSolver(QpSolver&&) noexcept;
  QpSolver& operator=(QpSolver&&) noexcept;
  ~QpSolver();

  // x 为热启动（清窗后为 0）。A 的有效行数 = n_ineq，0 ≤ n_ineq ≤ kMaxIneq。
  // 不等式语义与 Python 相同：A x >= b。
  QpStatus solve_box_ineq(
      const Mat36& P,
      const Vec36& g,
      const Vec36& lb,
      const Vec36& ub,
      const MatIneq& A,
      const VecIneq& b,
      int n_ineq,
      Vec36& x);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class QpSolver18 {
 public:
  QpSolver18();
  QpSolver18(const QpSolver18&) = delete;
  QpSolver18& operator=(const QpSolver18&) = delete;
  QpSolver18(QpSolver18&&) noexcept;
  QpSolver18& operator=(QpSolver18&&) noexcept;
  ~QpSolver18();

  // 热启动抄 36 维：x=0 清对偶；非零只写 primal，WARM_START_WITH_PREVIOUS_RESULT。
  QpStatus solve_box_ineq(
      const Mat18& P,
      const Vec18& g,
      const Vec18& lb,
      const Vec18& ub,
      const MatIneq18& A,
      const VecIneq& b,
      int n_ineq,
      Vec18& x);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kmpc
