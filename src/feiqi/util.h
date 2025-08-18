#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Sparse>
#include "types.h"

namespace mschol {

class pde_problem
{
 public:
  virtual size_t dim() const = 0;
  virtual void LHS(const double *u, Eigen::SparseMatrix<double> &A) const = 0;
  virtual void RHS(const double *u, Eigen::VectorXd &b) const = 0;
};

void compute_lame_coeffs(const double Ym, const double Ev,
                         double &mu, double &lam) {
  mu = Ym/(2.0*(1.0+Ev));
  lam = Ym*Ev/((1.0+Ev)*(1.0-2.0*Ev));  
}

matd_t generate_random_material(const mati_t &tris, const std::string &name,
                                const double maxE, const double minE,
                                const double pr=0.45) {
  ASSERT(name.substr(0, 4) == "RAND");

  const int mod = std::stoi(name.substr(4));
  std::cout << "rand material mod=" << mod << std::endl;

  vecd_t max_lame(2), min_lame(2);
  compute_lame_coeffs(maxE, pr, max_lame[0], max_lame[1]);
  compute_lame_coeffs(minE, pr, min_lame[0], min_lame[1]);

  matd_t mtr(2, tris.cols());  
  rand(); rand();
  for (size_t i = 0; i < mtr.cols(); ++i) {
    mtr.col(i) = (rand()%mod == 0) ? max_lame : min_lame;
  }

  return mtr;
}

}
#endif
