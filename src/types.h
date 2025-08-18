#ifndef NUMERIC_DEF_H
#define NUMERIC_DEF_H

#include <Eigen/Dense>

namespace mschol {

typedef Eigen::Matrix<size_t, -1, -1> mati_t;
typedef Eigen::Matrix<double, -1, -1> matd_t;
typedef Eigen::Matrix<size_t, -1,  1> veci_t;
typedef Eigen::Matrix<double, -1,  1> vecd_t;

}

namespace klchol {

typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> RmMatF_t;
typedef Eigen::Matrix<int,    -1, -1, Eigen::RowMajor> RmMatI_t;
typedef Eigen::Matrix<double, -1, -1, Eigen::ColMajor> CmMatF_t;
typedef Eigen::Matrix<int,    -1, -1, Eigen::ColMajor> CmMatI_t;

}
#endif // NUMERIC_DEF_H
