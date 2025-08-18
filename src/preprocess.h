#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <set>
#include <map>
// #include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <numeric>

#include "macro.h"
#include "types.h"
#include "maxmin.h"
#include "nanoflann.hpp"

namespace klchol {

class fps_sampler
{
 public:
  typedef KDTree kdtree_t;
  typedef double scalar_t;
  typedef int32_t index_t;
  
  fps_sampler(const RmMatF_t &pts) : pts_(pts), npts_(pts.rows())
  {
    // spdlog::info("number of samples={}", npts_);
    kdt_ = std::make_shared<kdtree_t>(3, std::cref(pts_), 10);
  }

  void compute(const char option, const kdtree_t *boundary=nullptr)
  {
    switch ( option ) {
      case 'F': fast_fps(boundary);        break;
      case 'B': brute_force_fps(boundary); break;
      default: exit(0);
    }
  }

  scalar_t get_len_scale(const index_t idx) const
  {
    return scale_[idx];
  }
  Eigen::Matrix<scalar_t, -1, 1> get_len_scale() const
  {
    return scale_;
  }

  const Eigen::PermutationMatrix<-1, -1>& P() const
  {
    return P_;
  }

  void reorder_geometry(RmMatF_t &V, RmMatI_t &F) const
  {
    for (index_t i = 0; i < F.rows(); ++i) {
      for (index_t j = 0; j < F.cols(); ++j) {
        F(i, j) = P_.indices()[F(i, j)];
      }
    }
    V.resize(pts_.rows(), pts_.cols());
    for (index_t i = 0; i < pts_.rows(); ++i) {
      V.row(P_.indices()[i]) = pts_.row(i);
    }
  }
  void reorder_geometry(RmMatF_t &V) const
  {
    V.resize(pts_.rows(), pts_.cols());
    #pragma omp parallel for
    for (index_t i = 0; i < pts_.rows(); ++i) {
      V.row(P_.indices()[i]) = pts_.row(i);
    }
  }

  void debug() const {
    // spdlog::info("length scale is acsending={}",std::is_sorted(scale_.data(), scale_.data()+scale_.size()));
    ASSERT(std::is_sorted(scale_.data(), scale_.data()+scale_.size()));
  }
  template <class Container>
  void debug(const Container &group) const
  {
    for (index_t i = 0; i < group.size()-1; ++i) {
      // spdlog::info("range=[{}, {}], is acsending={}", group[i], group[i+1], std::is_sorted(scale_.data()+group[i], scale_.data()+group[i+1]));
    }
  }

  void reset_all(const RmMatF_t &new_pts,
                 const Eigen::Matrix<scalar_t, -1, 1> &len_scale)
  {
    npts_ = new_pts.rows();
    pts_  = new_pts;
    scale_ = len_scale;

    kdt_.reset(new kdtree_t(3, std::cref(pts_), 10));

    P_.setIdentity(npts_);
    Pinv_.setIdentity(npts_);
  }

  void modify_length_scale(const std::function<void(scalar_t &)> &lambda)
  {
    std::cout << "# max length before modification=" << scale_.maxCoeff() << std::endl;
    std::cout << "# min length before modification=" << scale_.minCoeff() << std::endl;
    std::for_each(scale_.data(), scale_.data()+scale_.size(), lambda);
  }

  template <class SpMat, class Cont>
  void simpl_sparsity(const scalar_t rho,
                      const index_t dim,
                      const Cont &group,
                      const RmMatF_t &V,
                      SpMat &patt)
  {
    ASSERT(group.size() == 3 || group.size() == 2);
    
    typedef typename SpMat::Scalar Scalar;
    typedef typename SpMat::StorageIndex Index;
    // spdlog::info("sparsity pattern dim={}", dim);

    std::vector<std::vector<Index>> nei(dim*npts_);
    for (Index g = 0; g < group.size()-1; ++g) {
      const auto g_begin = group[g], g_end = group[g+1];

      // L_{pr, pr} and L_{tr, tr}      
      #pragma omp parallel for
      for (Index j = g_begin; j < g_end; ++j) {
        const auto J = Pinv_.indices()[j];
        const auto lj = rho*scale_[j];

        if ( j-g_begin < 0.7*(g_end-g_begin) ) {
          std::vector<std::pair<long int, scalar_t>> matches;
          const size_t nMatches = kdt_->index->radiusSearch(&pts_(J, 0), lj*lj, matches, nanoflann::SearchParams());
             
          for (const auto &res : matches) {
            const auto i = P_.indices()[res.first];

            // safe guard
            const auto I = res.first;
            const auto li = rho*scale_[i];
            if ( i >= j && i < g_end && (pts_.row(I)-pts_.row(J)).squaredNorm() < std::min(li*li, lj*lj) ) {

              for (Index pi = 0; pi < dim; ++pi) {
                for (Index pj = 0; pj < dim; ++pj) {
                  if ( dim*i+pi >= dim*j+pj ) {
                    nei[dim*j+pj].emplace_back(dim*i+pi);
                  }
                }
              }

            }
          }
        } else {
          for (Index i = j; i < g_end; ++i) {
            const auto I = Pinv_.indices()[i];
            const auto li = rho*scale_[i];
            if ( (pts_.row(I)-pts_.row(J)).squaredNorm() < std::min(li*li, lj*lj) ) {

              for (Index pi = 0; pi < dim; ++pi) {
                for (Index pj = 0; pj < dim; ++pj) {
                  if ( dim*i+pi >= dim*j+pj ) {
                    nei[dim*j+pj].emplace_back(dim*i+pi);
                  }
                }
              }

            }
          }
        }
      }
    }

    // L_{pr, tr} part
    if ( group.size() == 3 ) {
      const Index N_pr = group[1]-group[0], N_ob = group[2]-group[1];
      // spdlog::info("N_pr={}, N_ob={}", N_pr, N_ob);
      ASSERT(V.rows() == N_pr+N_ob);
      
      const RmMatF_t &V_ob = V.bottomRows(N_ob);
      kdtree_t kdt_ob(3, std::cref(V_ob), 10);

      // set scale the distance to observation for prediction points
      new_scale_.setZero(N_pr+N_ob);
      #pragma omp parallel for
      for (Index p = 0; p < N_pr; ++p) {
        scalar_t dist_to_ob = 0;
        size_t closest = 0;
        nanoflann::KNNResultSet<scalar_t> res(1);
        res.init(&closest, &dist_to_ob);
        kdt_ob.index->findNeighbors(res, &V(p, 0), nanoflann::SearchParams());
        new_scale_[p] = sqrt(dist_to_ob);
      }

      #pragma omp parallel for
      for (Index j = group[0]; j < group[1]; ++j) {
        // for every prediction point
        const auto lj = rho*new_scale_[j];
        std::vector<std::pair<long int, scalar_t>> matches;
        kdt_ob.index->radiusSearch(&V(j, 0), lj*lj, matches, nanoflann::SearchParams());
   
        for (const auto &res : matches) {          
          const auto i = res.first+N_pr;
          const auto li = rho*scale_[i];
          if ( (V.row(i)-V.row(j)).squaredNorm() < std::min(li*li, lj*lj) ) {
            
            for (Index pi = 0; pi < dim; ++pi) {
              for (Index pj = 0; pj < dim; ++pj) {
                if ( dim*i+pi >= dim*j+pj ) {
                  nei[dim*j+pj].emplace_back(dim*i+pi);
                }
              }
            }            
          }          
        }
      }
    }

    const Index n = dim*npts_;
    Eigen::Matrix<Index, -1, 1> L_ptr(n+1);
    L_ptr[0] = 0;
    for (Index i = 0; i < n; ++i) {
      L_ptr[i+1] = L_ptr[i]+nei[i].size();
    }

    const auto nnz = L_ptr[L_ptr.size()-1];
    Eigen::Matrix<Index,  -1, 1> L_ind(nnz);
    Eigen::Matrix<Scalar, -1, 1> L_val(nnz); L_val.setOnes();
    #pragma omp parallel for
    for (Index i = 0; i < nei.size(); ++i) {
      std::sort(nei[i].begin(), nei[i].end());
      std::copy(nei[i].begin(), nei[i].end(), L_ind.data()+L_ptr[i]);
    }    
    patt = Eigen::Map<SpMat>(n, n, nnz, L_ptr.data(), L_ind.data(), L_val.data());    
  }

  // MRA sparsity
  template <class SpMat>
  void simpl_sparsity(const scalar_t rho,
                      const index_t dim,
                      SpMat &patt) const
  {
    typedef typename SpMat::Scalar Scalar;
    typedef typename SpMat::StorageIndex Index;
    // spdlog::info("sparsity pattern dim={}", dim);

    scalar_t cutoff = 0.5;
    if ( rho <= 2.0 ) {
      cutoff = 0.95;
    } else if ( rho >= 7 ) {
      cutoff = 0.75;
    } else {
      cutoff = -0.04*rho+1.03;
    }

    std::vector<std::vector<Index>> nei(dim*npts_);
    #pragma omp parallel for
    for (Index j = 0; j < npts_; ++j) {
      const auto J = Pinv_.indices()[j];
      const auto lj = rho*scale_[j];

      if ( 1.0*j < cutoff*npts_ ) {
        std::vector<std::pair<long int, scalar_t>> matches;
        const size_t nMatches = kdt_->index->radiusSearch(&pts_(J, 0), lj*lj, matches, nanoflann::SearchParams());
        for (const auto &res : matches) {
          const auto i = P_.indices()[res.first];

          // safe guard
          const auto I = res.first;
          const auto li = rho*scale_[i];
          if ( i >= j && (pts_.row(I)-pts_.row(J)).squaredNorm() < std::min(li*li, lj*lj) ) {

            for (Index pi = 0; pi < dim; ++pi) {
              for (Index pj = 0; pj < dim; ++pj) {
                if ( dim*i+pi >= dim*j+pj ) {
                  nei[dim*j+pj].emplace_back(dim*i+pi);
                }
              }
            }

          }
        }
      } else {
        for (Index i = j; i < npts_; ++i) {
          const auto I = Pinv_.indices()[i];
          const auto li = rho*scale_[i];
          if ( (pts_.row(I)-pts_.row(J)).squaredNorm() < std::min(li*li, lj*lj) ) {
            //               || scale_[j] > 1e6 || scale_[i] > 1e6 ) {

            for (Index pi = 0; pi < dim; ++pi) {
              for (Index pj = 0; pj < dim; ++pj) {
                if ( dim*i+pi >= dim*j+pj ) {
                  nei[dim*j+pj].emplace_back(dim*i+pi);
                }
              }
            }

          }
        }
      }
    }

    const Index n = dim*npts_;
    Eigen::Matrix<Index, -1, 1> L_ptr(n+1);
    L_ptr[0] = 0;
    for (Index i = 0; i < n; ++i) {
      L_ptr[i+1] = L_ptr[i]+nei[i].size();
    }

    const auto nnz = L_ptr[L_ptr.size()-1];
    Eigen::Matrix<Index,  -1, 1> L_ind(nnz);
    Eigen::Matrix<Scalar, -1, 1> L_val(nnz); L_val.setOnes();
    #pragma omp parallel for
    for (Index i = 0; i < nei.size(); ++i) {
      std::sort(nei[i].begin(), nei[i].end());
      std::copy(nei[i].begin(), nei[i].end(), L_ind.data()+L_ptr[i]);
    }    
    patt = Eigen::Map<SpMat>(n, n, nnz, L_ptr.data(), L_ind.data(), L_val.data());
  }

  // nearest neighbor sparsity
  template <class SpMat>
  void nearest_sparsity(const int K,
                        const index_t dim,
                        SpMat &patt) const
  {
    typedef typename SpMat::Scalar Scalar;
    typedef typename SpMat::StorageIndex Index;
    // spdlog::info("sparsity pattern dim={}", dim);
    // spdlog::info("K neibs={}", K);

    Eigen::VectorXi processed = Eigen::VectorXi::Zero(npts_);
    std::vector<std::vector<Index>> nei(dim*npts_);

    const auto &sort_indices =
        [](const std::vector<scalar_t> &dist, const index_t offset,
           std::vector<index_t> &idx)
        {
          std::sort(idx.begin(), idx.end(),
                    [&](const index_t i1, const index_t i2) {
                      return dist[i1-offset] < dist[i2-offset];
                    });
        };
    
    // start from 2K nearest neighbors
    int curr_k = std::min(2*K, npts_);
    while ( processed.sum() < npts_ ) {
      std::cout << "processed=" << processed.sum() << std::endl;

      #pragma omp parallel for
      for (Index j = 0; j < processed.size(); ++j) {
        if ( processed[j] == 0 ) {
          const auto J = Pinv_.indices()[j];

          const index_t totalJ = npts_-j;
          if ( totalJ <= curr_k ) {
            processed[j] = 1;

            std::vector<scalar_t> dist(totalJ);
            for (index_t i = j; i < npts_; ++i) {
              const auto I = Pinv_.indices()[i];
              dist[i-j] = (pts_.row(I)-pts_.row(J)).squaredNorm();
            }
            std::vector<index_t> idx(totalJ);
            std::iota(idx.begin(), idx.end(), j);
            sort_indices(dist, j, idx);
            
            for (index_t w = 0; w < std::min(K, totalJ); ++w) {
              const index_t i = idx[w];
              for (Index pi = 0; pi < dim; ++pi) {
                for (Index pj = 0; pj < dim; ++pj) {
                  if ( dim*i+pi >= dim*j+pj ) {
                    nei[dim*j+pj].emplace_back(dim*i+pi);
                  }
                }
              }
            }
          } else {
            std::vector<scalar_t> dist(curr_k);
            std::vector<size_t> indices(curr_k);
            nanoflann::KNNResultSet<scalar_t> res(curr_k);
            res.init(&indices[0], &dist[0]);
            kdt_->index->findNeighbors(res, &pts_(J, 0), nanoflann::SearchParams());

            // locate valid neighbors
            std::vector<size_t> valid_nei;
            for (const auto &pid : indices) {
              const index_t i = P_.indices()[pid];
              if ( i >= j ) {
                valid_nei.emplace_back(i);
                if ( valid_nei.size() == K ) {
                  break;
                }                
              }
            }
            if ( valid_nei.size() == K ) {
              processed[j] = 1;

              for (const auto i : valid_nei) {
                for (Index pi = 0; pi < dim; ++pi) {
                  for (Index pj = 0; pj < dim; ++pj) {
                    if ( dim*i+pi >= dim*j+pj ) {
                      nei[dim*j+pj].emplace_back(dim*i+pi);
                    }
                  }
                }
              }
            }
          }
        }
      }

      curr_k = std::min(2*curr_k, npts_);      
    }

    const Index n = dim*npts_;
    Eigen::Matrix<Index, -1, 1> L_ptr(n+1);
    L_ptr[0] = 0;
    for (Index i = 0; i < n; ++i) {
      L_ptr[i+1] = L_ptr[i]+nei[i].size();
    }

    const auto nnz = L_ptr[L_ptr.size()-1];
    Eigen::Matrix<Index,  -1, 1> L_ind(nnz);
    Eigen::Matrix<Scalar, -1, 1> L_val(nnz); L_val.setOnes();
    #pragma omp parallel for
    for (Index i = 0; i < nei.size(); ++i) {
      std::sort(nei[i].begin(), nei[i].end());
      std::copy(nei[i].begin(), nei[i].end(), L_ind.data()+L_ptr[i]);
    }    
    patt = Eigen::Map<SpMat>(n, n, nnz, L_ptr.data(), L_ind.data(), L_val.data());    
  }

  template <class SpMat, class VecI, class Cont>
  void nearest_aggregate(const index_t  dim,
                         const Cont     &group,
                         const SpMat    &patt,
                         const scalar_t lambda, 
                         VecI         &sup_ptr,
                         VecI         &sup_ind,
                         VecI         &sup_parent,
                         const index_t max_super_size=INT_MAX) const
  {
    typedef typename VecI::Scalar INT;

    const index_t N = patt.rows()/dim;
    ASSERT(N == npts_);

    std::vector<std::vector<index_t>> neib(N);
    #pragma omp parallel for
    for (index_t j = 0; j < N; ++j) { // for each point
      const index_t nnz_j = (patt.outerIndexPtr()[j*dim+1]-patt.outerIndexPtr()[j*dim])/dim;
      neib[j].resize(nnz_j);
      index_t cnt_j = 0;
      for (auto iter = patt.outerIndexPtr()[j*dim]; iter < patt.outerIndexPtr()[j*dim+1]; iter += dim) {
        const index_t i = patt.innerIndexPtr()[iter]/dim;
        neib[j][cnt_j++] = i;
      }
    }

    // init a find-union set
    std::vector<index_t> parent(N);
    for (index_t i = 0; i < N; ++i) {
      parent[i] = i;
    }
    
    const auto &Find =
        [&](const index_t elem_i) {
          index_t l = elem_i;
          while ( parent[l] != l ) {
            l = parent[l];
          }
          return l;
        };
    const auto &Merge =
        [&](const index_t rep_i, const index_t rep_j) { // merge j to i
          parent[rep_j] = rep_i;
        };
    const auto &Compress =
        [&](std::vector<index_t> &parent) {
          std::vector<index_t> new_parent(parent.size());
          #pragma omp parallel for
          for (index_t i = 0; i < new_parent.size(); ++i) {
            new_parent[i] = Find(i);
          }
          std::copy(new_parent.begin(), new_parent.end(), parent.begin());
        };

    for (index_t j = 0; j < N; ++j) {
      const index_t rep_j = Find(j);
      const index_t n_Uj = neib[rep_j].size();
      
      for (auto iter = patt.outerIndexPtr()[j*dim]; iter < patt.outerIndexPtr()[j*dim+1]; iter += dim) {
        const index_t i = patt.innerIndexPtr()[iter]/dim;
        const index_t rep_i = Find(i);

        // same class
        if ( rep_i == rep_j ) continue;

        const index_t n_Ui = neib[rep_i].size();
        std::vector<index_t> merged_Uij;
        std::set_union(neib[rep_i].cbegin(), neib[rep_i].cend(),
                       neib[rep_j].cbegin(), neib[rep_j].cend(),
                       std::back_inserter(merged_Uij));
        const index_t n_Ui_Uj = merged_Uij.size();

        // merge two sets        
        if ( n_Ui_Uj*n_Ui_Uj <= lambda*(n_Ui*n_Ui+n_Uj*n_Uj) ) {
          Merge(rep_i, rep_j);
          neib[rep_i] = merged_Uij;
          neib[rep_j].clear();
        }
      }
    }

    Compress(parent);
    std::map<index_t, index_t> p_map;
    index_t n_super = 0;
    for (const auto &pa : parent) {
      if ( p_map.find(pa) == p_map.end() ) {
        p_map.insert(std::make_pair(pa, n_super++));
      }
    }
    // spdlog::info("nearest n_super={}", n_super);
    sup_ptr.resize(n_super+1);
    sup_ind.resize(N);
    sup_parent.resize(N);

    std::vector<std::vector<index_t>> sup(n_super);
    for (index_t k = 0; k < N; ++k) {
      const auto p_id = p_map[parent[k]];
      sup_parent[k] = p_id;
      sup[p_id].emplace_back(k);
    }

    sup_ptr[0] = 0;
    for (index_t i = 1; i < sup_ptr.size(); ++i) {
      sup_ptr[i] = sup_ptr[i-1]+sup[i-1].size();
      std::copy(sup[i-1].begin(), sup[i-1].end(), &sup_ind[sup_ptr[i-1]]);
    }
    ASSERT(N == sup_ptr[sup_ptr.size()-1]);
  }
  
  template <class SpMat, class VecI, class Cont>
  void aggregate(const index_t  dim,
                 const Cont     &group,
                 const SpMat    &patt,
                 const scalar_t lambda, 
                 VecI         &sup_ptr,
                 VecI         &sup_ind,
                 VecI         &sup_parent,
                 const index_t max_super_size=INT_MAX) const
  {
    typedef typename VecI::Scalar INT;

    const index_t N = patt.rows()/dim;
    ASSERT(N == npts_);
    std::vector<unsigned char> vis(N, 0);

    std::vector<std::vector<INT>> sup;
    for (index_t g = 0; g < group.size()-1; ++g) {
      const auto g_begin = group[g], g_end = group[g+1];

      for (index_t j = g_begin; j < g_end; ++j) { // for each node
        if ( vis[j] ) { // has been aggregated into a supernode
          continue;
        }
        
        // start a new supernode
        sup.emplace_back(std::vector<INT>());
        for (typename SpMat::InnerIterator it(patt, dim*j); it; ++it) {
          const index_t i = it.row()/dim;
          if ( i < g_begin ) continue;
          if ( i >= g_end ) break;
          if ( !vis[i] && scale_[i] <= lambda*scale_[j] ) {
            vis[i] = 1;
            sup.back().emplace_back(i);
            if ( sup.back().size() >= max_super_size ) {
              break;
            }
          }
        }
      }
    }
    const index_t sup_size = sup.size();
    // spdlog::info("# supernodes={}", sup_size);
    
    // store supernodes indices
    sup_ptr.resize(sup_size+1);
    sup_ind.resize(N);
    sup_ptr[0] = 0;
    for (index_t i = 1; i < sup_ptr.size(); ++i) {
      sup_ptr[i] = sup_ptr[i-1]+sup[i-1].size();
      std::copy(sup[i-1].begin(), sup[i-1].end(), &sup_ind[sup_ptr[i-1]]);
    }
    ASSERT(N == sup_ptr[sup_ptr.size()-1]);    

    // build parent mapping
    sup_parent.resize(N);
    for (index_t i = 0; i < sup.size(); ++i) {
      for (const auto idx : sup[i]) {
        sup_parent[idx] = i;
      }
    }
  }

  template <class SpMat, class VecI> 
  void super_sparsity(const index_t dim,
                      const SpMat &patt,
                      const VecI  &sup_parent,
                      SpMat       &sup_patt)
  {
    typedef typename SpMat::Scalar Scalar;
    typedef typename SpMat::StorageIndex Index;
    
    const index_t N = patt.rows()/dim;
    const index_t n_super = sup_parent.maxCoeff()+1;
    const SpMat &PT = patt.transpose();
    // spdlog::info("[super sparsity] n_super={}", n_super);

    // find the union of row indices for each supernode
    std::vector<std::vector<Index>> idx_set(n_super);
    for (Index i = 0; i < PT.cols(); ++i) {
      for (typename SpMat::InnerIterator it(PT, i); it; ++it) {
        const Index J = it.row()/dim;
        const Index PARENT_I = sup_parent[J];
        idx_set[PARENT_I].emplace_back(i);
      }
    }

    // for (auto &vec : idx_set) {
    //   vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    // }
    #pragma omp parallel for
    for (Index i = 0; i < idx_set.size(); ++i) {
      auto &vec = idx_set[i];
      vec.erase(std::unique(vec.begin(), vec.end()), vec.end());      
    }

    const Index n_dof = patt.cols();
    Eigen::Matrix<Index, -1, 1> L_ptr(n_dof+1);
    L_ptr[0] = 0;
    // for (Index j = 0; j < n_dof; ++j) {
    //   const Index J = j/dim;
    //   const Index PJ = sup_parent[J];
    //   const auto it = std::lower_bound(idx_set[PJ].begin(), idx_set[PJ].end(), j);
    //   L_ptr[j+1] = L_ptr[j]+(idx_set[PJ].end()-it);
    // }
    std::vector<typename std::vector<Index>::iterator> offset(n_dof);
    #pragma omp parallel for
    for (Index j = 0; j < n_dof; ++j) {
      const Index J = j/dim;
      const Index PJ = sup_parent[J];
      offset[j] = std::lower_bound(idx_set[PJ].begin(), idx_set[PJ].end(), j); 
    }
    for (Index j = 0; j < n_dof; ++j) {
      const Index J = j/dim;
      const Index PJ = sup_parent[J];            
      L_ptr[j+1] = L_ptr[j]+(idx_set[PJ].end()-offset[j]);
    }
    
    const Index nnz = L_ptr[L_ptr.size()-1];
    Eigen::Matrix<Index, -1, 1> L_ind(nnz);
    //    #pragma omp parallel for
    for (Index j = 0; j < n_dof; ++j) {
      const Index J = j/dim;
      const Index PJ = sup_parent[J];
      // const auto it = std::lower_bound(idx_set[PJ].begin(), idx_set[PJ].end(), j);
      std::copy(offset[j], idx_set[PJ].end(), L_ind.data()+L_ptr[j]);
    }

    Eigen::Matrix<Scalar, -1, 1> L_val(nnz); L_val.setOnes();
    sup_patt = Eigen::Map<SpMat>(n_dof, n_dof, nnz, L_ptr.data(), L_ind.data(), L_val.data());
  }  
  
 private:
  index_t npts_;
  RmMatF_t pts_;
  Eigen::Matrix<scalar_t, -1, 1> scale_, new_scale_;

  std::shared_ptr<kdtree_t> kdt_;

  // fine-to-coarse!
  Eigen::PermutationMatrix<-1, -1> P_, Pinv_;

 private:
  void brute_force_fps(const kdtree_t *boundary) {
    const index_t N = pts_.rows();
    scale_.resize(N);
    
    std::vector<index_t> seq;
    std::vector<unsigned char> vis(N, 0);
    {
      // select the first point
      const Eigen::RowVector3d &corner = pts_.colwise().minCoeff();
      // spdlog::info("corner ({0:.8f}, {1:.8f}, {2:.8f})", corner.x(), corner.y(), corner.z());
      
      index_t first_idx = -1;
      if ( boundary ) {
        std::cout << "# boundary is present!" << std::endl;
        // select the farthest one to the boundary
        scalar_t dist_to_corner = 0;
        for (index_t i = 0; i < npts_; ++i) {
          size_t ret_index;
          double dist;
          nanoflann::KNNResultSet<double> resultSet(1);
          resultSet.init(&ret_index, &dist);
          boundary->index->findNeighbors(resultSet, &pts_(i, 0), nanoflann::SearchParams());
          if ( dist > dist_to_corner ) {
            dist_to_corner = dist;
            first_idx = i;
          }
        }
      } else {
        std::cout << "# boundary is NOT present!" << std::endl;
        // select the closest to the domain corner
        scalar_t dist_to_corner = 1e10;
        for (index_t i = 0; i < N; ++i) {
          scalar_t dist = (pts_.row(i)-corner).squaredNorm();
          if ( dist < dist_to_corner ) {
            dist_to_corner = dist;
            first_idx = i;
          }
        }
      }

      seq.emplace_back(first_idx);
      vis[first_idx] = 1;
      scale_[first_idx] = 1e10; // the first one has inf length scale
    }
    
    while ( seq.size() < N ) {
      if ( seq.size()%100 == 0 ) {
        std::cout << "processed " << seq.size() << " points" << std::endl;
      }
      
      #pragma omp parallel for
      for (index_t i = 0; i < N; ++i) {
        if ( !vis[i] ) {
          scalar_t dist_to_cluster = 1e10;
          for (index_t j = 0; j < seq.size(); ++j) {
            scalar_t dist = (pts_.row(i)-pts_.row(seq[j])).squaredNorm();
            if ( dist < dist_to_cluster ) {
              dist_to_cluster = dist;
            }
          }

          double dist_to_boundary = 1e20;
          if ( boundary ) {
            size_t ret_index;            
            nanoflann::KNNResultSet<double> resultSet(1);
            resultSet.init(&ret_index, &dist_to_boundary);
            boundary->index->findNeighbors(resultSet, &pts_(i, 0), nanoflann::SearchParams());
          }

          scale_[i] = std::sqrt(std::min(dist_to_cluster, dist_to_boundary));
        }
      }

      // find the farthest one
      scalar_t max_dist = 0;
      index_t max_idx = -1;
      for (index_t i = 0; i < N; ++i) {
        if ( !vis[i] ) {
          if ( scale_[i] > max_dist ) {
            max_dist = scale_[i];
            max_idx = i;
          }
        }
      }
      vis[max_idx] = 1;
      seq.emplace_back(max_idx);
    }

    // the coarsest goes to the last
    P_.resize(N);
    for (index_t i = 0; i < N; ++i) {
      P_.indices()[seq[i]] = N-1-i;
    }
    Pinv_ = P_.inverse();
    scale_ = P_*scale_;
    std::cout << "first length=" << scale_.head(1) << std::endl;
    std::cout << "last length=" << scale_.tail(1) << std::endl;
  }
  void fast_fps(const kdtree_t *boundary) {
    ASSERT(pts_.rows() == npts_);
    
    // select the first point
    const Eigen::RowVector3d &corner = pts_.colwise().minCoeff();
    // spdlog::info("corner ({0:.8f}, {1:.8f}, {2:.8f})", corner.x(), corner.y(), corner.z());
      
    index_t first_idx = -1;
    if ( boundary ) {
      // there is a boundary, select the farthest to the boundary
      scalar_t dist_to_corner = 0;
      for (index_t i = 0; i < npts_; ++i) {
        size_t ret_index;
        double dist;
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index, &dist);
        boundary->index->findNeighbors(resultSet, &pts_(i, 0), nanoflann::SearchParams());
        if ( dist > dist_to_corner ) {
          dist_to_corner = dist;
          first_idx = i;
        }
      }
    } else {
      // select the closest to the boundary
      scalar_t dist_to_corner = 1e10;      
      for (index_t i = 0; i < npts_; ++i) {
        scalar_t dist = (pts_.row(i)-corner).squaredNorm();
        if ( dist < dist_to_corner ) {
          dist_to_corner = dist;
          first_idx = i;
        }
      }
    }
    // spdlog::info("first_idx={}", first_idx);
    
    std::vector<unsigned int> seq(npts_), rev_seq(npts_);
    scale_.resize(npts_);
    seq[0] = first_idx;
    create_ordering_3d(&seq[0], &rev_seq[0], scale_.data(), npts_, pts_.data(), first_idx, boundary);

    std::set<unsigned int> seq_set(seq.begin(), seq.end());
    ASSERT(seq_set.size() == npts_);

    P_.resize(npts_);
    for (index_t i = 0; i < npts_; ++i) {
      P_.indices()[seq[i]] = npts_-1-i;
    }
    Pinv_ = P_.inverse();
    scale_.reverseInPlace();
  }
};

// class fps_sampler_float
// {
//  public:
//   typedef KDTree kdtree_t;
//   typedef float scalar_t;
//   typedef int32_t index_t;
//   typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> RmMatF_float_t;
  
//   fps_sampler_float(const RmMatF_float_t &pts) : pts_(pts), npts_(pts.rows())
//   {
//     // spdlog::info("number of samples={}", npts_);
//     kdt_ = std::make_shared<kdtree_t>(3, std::cref(pts_), 10);
//   }

//   void compute(const char option, const kdtree_t *boundary=nullptr)
//   {
//     switch ( option ) {
//       case 'F': fast_fps(boundary);        break;
//       case 'B': brute_force_fps(boundary); break;
//       default: exit(0);
//     }
//   }

//   scalar_t get_len_scale(const index_t idx) const
//   {
//     return scale_[idx];
//   }
//   Eigen::Matrix<scalar_t, -1, 1> get_len_scale() const
//   {
//     return scale_;
//   }

//   const Eigen::PermutationMatrix<-1, -1>& P() const
//   {
//     return P_;
//   }

//   void reorder_geometry(RmMatF_float_t &V, RmMatI_t &F) const
//   {
//     for (index_t i = 0; i < F.rows(); ++i) {
//       for (index_t j = 0; j < F.cols(); ++j) {
//         F(i, j) = P_.indices()[F(i, j)];
//       }
//     }
//     V.resize(pts_.rows(), pts_.cols());
//     for (index_t i = 0; i < pts_.rows(); ++i) {
//       V.row(P_.indices()[i]) = pts_.row(i);
//     }
//   }
//   void reorder_geometry(RmMatF_float_t &V) const
//   {
//     V.resize(pts_.rows(), pts_.cols());
//     #pragma omp parallel for
//     for (index_t i = 0; i < pts_.rows(); ++i) {
//       V.row(P_.indices()[i]) = pts_.row(i);
//     }
//   }

//   void debug() const {
//     // spdlog::info("length scale is acsending={}",std::is_sorted(scale_.data(), scale_.data()+scale_.size()));
//     ASSERT(std::is_sorted(scale_.data(), scale_.data()+scale_.size()));
//   }
//   template <class Container>
//   void debug(const Container &group) const
//   {
//     for (index_t i = 0; i < group.size()-1; ++i) {
//       spdlog::info("range=[{}, {}], is acsending={}", group[i], group[i+1], std::is_sorted(scale_.data()+group[i], scale_.data()+group[i+1]));
//     }
//   }

//   void reset_all(const RmMatF_float_t &new_pts,
//                  const Eigen::Matrix<scalar_t, -1, 1> &len_scale)
//   {
//     npts_ = new_pts.rows();
//     pts_  = new_pts;
//     scale_ = len_scale;

//     kdt_.reset(new kdtree_t(3, std::cref(pts_), 10));

//     P_.setIdentity(npts_);
//     Pinv_.setIdentity(npts_);
//   }

//   void modify_length_scale(const std::function<void(scalar_t &)> &lambda)
//   {
//     std::cout << "# max length before modification=" << scale_.maxCoeff() << std::endl;
//     std::cout << "# min length before modification=" << scale_.minCoeff() << std::endl;
//     std::for_each(scale_.data(), scale_.data()+scale_.size(), lambda);
//   }

//   template <class SpMat, class Cont>
//   void simpl_sparsity(const scalar_t rho,
//                       const index_t dim,
//                       const Cont &group,
//                       const RmMatF_float_t &V,
//                       SpMat &patt)
//   {
//     ASSERT(group.size() == 3 || group.size() == 2);
    
//     typedef typename SpMat::Scalar Scalar;
//     typedef typename SpMat::StorageIndex Index;
//     spdlog::info("sparsity pattern dim={}", dim);

//     std::vector<std::vector<Index>> nei(dim*npts_);
//     for (Index g = 0; g < group.size()-1; ++g) {
//       const auto g_begin = group[g], g_end = group[g+1];

//       // L_{pr, pr} and L_{tr, tr}      
//       #pragma omp parallel for
//       for (Index j = g_begin; j < g_end; ++j) {
//         const auto J = Pinv_.indices()[j];
//         const auto lj = rho*scale_[j];

//         if ( j-g_begin < 0.7*(g_end-g_begin) ) {
//           std::vector<std::pair<long int, scalar_t>> matches;
//           const size_t nMatches = kdt_->index->radiusSearch(&pts_(J, 0), lj*lj, matches, nanoflann::SearchParams());
             
//           for (const auto &res : matches) {
//             const auto i = P_.indices()[res.first];

//             // safe guard
//             const auto I = res.first;
//             const auto li = rho*scale_[i];
//             if ( i >= j && i < g_end && (pts_.row(I)-pts_.row(J)).squaredNorm() < std::min(li*li, lj*lj) ) {

//               for (Index pi = 0; pi < dim; ++pi) {
//                 for (Index pj = 0; pj < dim; ++pj) {
//                   if ( dim*i+pi >= dim*j+pj ) {
//                     nei[dim*j+pj].emplace_back(dim*i+pi);
//                   }
//                 }
//               }

//             }
//           }
//         } else {
//           for (Index i = j; i < g_end; ++i) {
//             const auto I = Pinv_.indices()[i];
//             const auto li = rho*scale_[i];
//             if ( (pts_.row(I)-pts_.row(J)).squaredNorm() < std::min(li*li, lj*lj) ) {

//               for (Index pi = 0; pi < dim; ++pi) {
//                 for (Index pj = 0; pj < dim; ++pj) {
//                   if ( dim*i+pi >= dim*j+pj ) {
//                     nei[dim*j+pj].emplace_back(dim*i+pi);
//                   }
//                 }
//               }

//             }
//           }
//         }
//       }
//     }

//     // L_{pr, tr} part
//     if ( group.size() == 3 ) {
//       const Index N_pr = group[1]-group[0], N_ob = group[2]-group[1];
//       spdlog::info("N_pr={}, N_ob={}", N_pr, N_ob);
//       ASSERT(V.rows() == N_pr+N_ob);
      
//       const RmMatF_float_t &V_ob = V.bottomRows(N_ob);
//       kdtree_t kdt_ob(3, std::cref(V_ob), 10);

//       // set scale the distance to observation for prediction points
//       new_scale_.setZero(N_pr+N_ob);
//       #pragma omp parallel for
//       for (Index p = 0; p < N_pr; ++p) {
//         scalar_t dist_to_ob = 0;
//         size_t closest = 0;
//         nanoflann::KNNResultSet<scalar_t> res(1);
//         res.init(&closest, &dist_to_ob);
//         kdt_ob.index->findNeighbors(res, &V(p, 0), nanoflann::SearchParams());
//         new_scale_[p] = sqrtf(dist_to_ob);
//       }

//       #pragma omp parallel for
//       for (Index j = group[0]; j < group[1]; ++j) {
//         // for every prediction point
//         const auto lj = rho*new_scale_[j];
//         std::vector<std::pair<long int, scalar_t>> matches;
//         kdt_ob.index->radiusSearch(&V(j, 0), lj*lj, matches, nanoflann::SearchParams());
   
//         for (const auto &res : matches) {          
//           const auto i = res.first+N_pr;
//           const auto li = rho*scale_[i];
//           if ( (V.row(i)-V.row(j)).squaredNorm() < std::min(li*li, lj*lj) ) {
            
//             for (Index pi = 0; pi < dim; ++pi) {
//               for (Index pj = 0; pj < dim; ++pj) {
//                 if ( dim*i+pi >= dim*j+pj ) {
//                   nei[dim*j+pj].emplace_back(dim*i+pi);
//                 }
//               }
//             }            
//           }          
//         }
//       }
//     }

//     const Index n = dim*npts_;
//     Eigen::Matrix<Index, -1, 1> L_ptr(n+1);
//     L_ptr[0] = 0;
//     for (Index i = 0; i < n; ++i) {
//       L_ptr[i+1] = L_ptr[i]+nei[i].size();
//     }

//     const auto nnz = L_ptr[L_ptr.size()-1];
//     Eigen::Matrix<Index,  -1, 1> L_ind(nnz);
//     Eigen::Matrix<Scalar, -1, 1> L_val(nnz); L_val.setOnes();
//     #pragma omp parallel for
//     for (Index i = 0; i < nei.size(); ++i) {
//       std::sort(nei[i].begin(), nei[i].end());
//       std::copy(nei[i].begin(), nei[i].end(), L_ind.data()+L_ptr[i]);
//     }    
//     patt = Eigen::Map<SpMat>(n, n, nnz, L_ptr.data(), L_ind.data(), L_val.data());    
//   }

//   // MRA sparsity
//   template <class SpMat>
//   void simpl_sparsity(const scalar_t rho,
//                       const index_t dim,
//                       SpMat &patt) const
//   {
//     typedef typename SpMat::Scalar Scalar;
//     typedef typename SpMat::StorageIndex Index;
//     // spdlog::info("sparsity pattern dim={}", dim);

//     scalar_t cutoff = 0.5f;
//     if ( rho <= 2.0f ) {
//       cutoff = 0.95f;
//     } else if ( rho >= 7.0f ) {
//       cutoff = 0.75f;
//     } else {
//       cutoff = -0.04f*rho+1.03f;
//     }

//     std::vector<std::vector<Index>> nei(dim*npts_);
//     #pragma omp parallel for
//     for (Index j = 0; j < npts_; ++j) {
//       const auto J = Pinv_.indices()[j];
//       const auto lj = rho*scale_[j];

//       if ( 1.0f*j < cutoff*npts_ ) {
//         std::vector<std::pair<long int, scalar_t>> matches;
//         const size_t nMatches = kdt_->index->radiusSearch(&pts_(J, 0), lj*lj, matches, nanoflann::SearchParams());
//         for (const auto &res : matches) {
//           const auto i = P_.indices()[res.first];

//           // safe guard
//           const auto I = res.first;
//           const auto li = rho*scale_[i];
//           if ( i >= j && (pts_.row(I)-pts_.row(J)).squaredNorm() < std::min(li*li, lj*lj) ) {

//             for (Index pi = 0; pi < dim; ++pi) {
//               for (Index pj = 0; pj < dim; ++pj) {
//                 if ( dim*i+pi >= dim*j+pj ) {
//                   nei[dim*j+pj].emplace_back(dim*i+pi);
//                 }
//               }
//             }

//           }
//         }
//       } else {
//         for (Index i = j; i < npts_; ++i) {
//           const auto I = Pinv_.indices()[i];
//           const auto li = rho*scale_[i];
//           if ( (pts_.row(I)-pts_.row(J)).squaredNorm() < std::min(li*li, lj*lj) ) {
//             //               || scale_[j] > 1e6 || scale_[i] > 1e6 ) {

//             for (Index pi = 0; pi < dim; ++pi) {
//               for (Index pj = 0; pj < dim; ++pj) {
//                 if ( dim*i+pi >= dim*j+pj ) {
//                   nei[dim*j+pj].emplace_back(dim*i+pi);
//                 }
//               }
//             }

//           }
//         }
//       }
//     }

//     const Index n = dim*npts_;
//     Eigen::Matrix<Index, -1, 1> L_ptr(n+1);
//     L_ptr[0] = 0;
//     for (Index i = 0; i < n; ++i) {
//       L_ptr[i+1] = L_ptr[i]+nei[i].size();
//     }

//     const auto nnz = L_ptr[L_ptr.size()-1];
//     Eigen::Matrix<Index,  -1, 1> L_ind(nnz);
//     Eigen::Matrix<Scalar, -1, 1> L_val(nnz); L_val.setOnes();
//     #pragma omp parallel for
//     for (Index i = 0; i < nei.size(); ++i) {
//       std::sort(nei[i].begin(), nei[i].end());
//       std::copy(nei[i].begin(), nei[i].end(), L_ind.data()+L_ptr[i]);
//     }    
//     patt = Eigen::Map<SpMat>(n, n, nnz, L_ptr.data(), L_ind.data(), L_val.data());
//   }

//   // nearest neighbor sparsity
//   template <class SpMat>
//   void nearest_sparsity(const int K,
//                         const index_t dim,
//                         SpMat &patt) const
//   {
//     typedef typename SpMat::Scalar Scalar;
//     typedef typename SpMat::StorageIndex Index;
//     spdlog::info("sparsity pattern dim={}", dim);
//     spdlog::info("K neibs={}", K);

//     Eigen::VectorXi processed = Eigen::VectorXi::Zero(npts_);
//     std::vector<std::vector<Index>> nei(dim*npts_);

//     const auto &sort_indices =
//         [](const std::vector<scalar_t> &dist, const index_t offset,
//            std::vector<index_t> &idx)
//         {
//           std::sort(idx.begin(), idx.end(),
//                     [&](const index_t i1, const index_t i2) {
//                       return dist[i1-offset] < dist[i2-offset];
//                     });
//         };
    
//     // start from 2K nearest neighbors
//     int curr_k = std::min(2*K, npts_);
//     while ( processed.sum() < npts_ ) {
//       std::cout << "processed=" << processed.sum() << std::endl;

//       #pragma omp parallel for
//       for (Index j = 0; j < processed.size(); ++j) {
//         if ( processed[j] == 0 ) {
//           const auto J = Pinv_.indices()[j];

//           const index_t totalJ = npts_-j;
//           if ( totalJ <= curr_k ) {
//             processed[j] = 1;

//             std::vector<scalar_t> dist(totalJ);
//             for (index_t i = j; i < npts_; ++i) {
//               const auto I = Pinv_.indices()[i];
//               dist[i-j] = (pts_.row(I)-pts_.row(J)).squaredNorm();
//             }
//             std::vector<index_t> idx(totalJ);
//             std::iota(idx.begin(), idx.end(), j);
//             sort_indices(dist, j, idx);
            
//             for (index_t w = 0; w < std::min(K, totalJ); ++w) {
//               const index_t i = idx[w];
//               for (Index pi = 0; pi < dim; ++pi) {
//                 for (Index pj = 0; pj < dim; ++pj) {
//                   if ( dim*i+pi >= dim*j+pj ) {
//                     nei[dim*j+pj].emplace_back(dim*i+pi);
//                   }
//                 }
//               }
//             }
//           } else {
//             std::vector<scalar_t> dist(curr_k);
//             std::vector<size_t> indices(curr_k);
//             nanoflann::KNNResultSet<scalar_t> res(curr_k);
//             res.init(&indices[0], &dist[0]);
//             kdt_->index->findNeighbors(res, &pts_(J, 0), nanoflann::SearchParams());

//             // locate valid neighbors
//             std::vector<size_t> valid_nei;
//             for (const auto &pid : indices) {
//               const index_t i = P_.indices()[pid];
//               if ( i >= j ) {
//                 valid_nei.emplace_back(i);
//                 if ( valid_nei.size() == K ) {
//                   break;
//                 }                
//               }
//             }
//             if ( valid_nei.size() == K ) {
//               processed[j] = 1;

//               for (const auto i : valid_nei) {
//                 for (Index pi = 0; pi < dim; ++pi) {
//                   for (Index pj = 0; pj < dim; ++pj) {
//                     if ( dim*i+pi >= dim*j+pj ) {
//                       nei[dim*j+pj].emplace_back(dim*i+pi);
//                     }
//                   }
//                 }
//               }
//             }
//           }
//         }
//       }

//       curr_k = std::min(2*curr_k, npts_);      
//     }

//     const Index n = dim*npts_;
//     Eigen::Matrix<Index, -1, 1> L_ptr(n+1);
//     L_ptr[0] = 0;
//     for (Index i = 0; i < n; ++i) {
//       L_ptr[i+1] = L_ptr[i]+nei[i].size();
//     }

//     const auto nnz = L_ptr[L_ptr.size()-1];
//     Eigen::Matrix<Index,  -1, 1> L_ind(nnz);
//     Eigen::Matrix<Scalar, -1, 1> L_val(nnz); L_val.setOnes();
//     #pragma omp parallel for
//     for (Index i = 0; i < nei.size(); ++i) {
//       std::sort(nei[i].begin(), nei[i].end());
//       std::copy(nei[i].begin(), nei[i].end(), L_ind.data()+L_ptr[i]);
//     }    
//     patt = Eigen::Map<SpMat>(n, n, nnz, L_ptr.data(), L_ind.data(), L_val.data());    
//   }

//   template <class SpMat, class VecI, class Cont>
//   void nearest_aggregate(const index_t  dim,
//                          const Cont     &group,
//                          const SpMat    &patt,
//                          const scalar_t lambda, 
//                          VecI         &sup_ptr,
//                          VecI         &sup_ind,
//                          VecI         &sup_parent,
//                          const index_t max_super_size=INT_MAX) const
//   {
//     typedef typename VecI::Scalar INT;

//     const index_t N = patt.rows()/dim;
//     ASSERT(N == npts_);

//     std::vector<std::vector<index_t>> neib(N);
//     #pragma omp parallel for
//     for (index_t j = 0; j < N; ++j) { // for each point
//       const index_t nnz_j = (patt.outerIndexPtr()[j*dim+1]-patt.outerIndexPtr()[j*dim])/dim;
//       neib[j].resize(nnz_j);
//       index_t cnt_j = 0;
//       for (auto iter = patt.outerIndexPtr()[j*dim]; iter < patt.outerIndexPtr()[j*dim+1]; iter += dim) {
//         const index_t i = patt.innerIndexPtr()[iter]/dim;
//         neib[j][cnt_j++] = i;
//       }
//     }

//     // init a find-union set
//     std::vector<index_t> parent(N);
//     for (index_t i = 0; i < N; ++i) {
//       parent[i] = i;
//     }
    
//     const auto &Find =
//         [&](const index_t elem_i) {
//           index_t l = elem_i;
//           while ( parent[l] != l ) {
//             l = parent[l];
//           }
//           return l;
//         };
//     const auto &Merge =
//         [&](const index_t rep_i, const index_t rep_j) { // merge j to i
//           parent[rep_j] = rep_i;
//         };
//     const auto &Compress =
//         [&](std::vector<index_t> &parent) {
//           std::vector<index_t> new_parent(parent.size());
//           #pragma omp parallel for
//           for (index_t i = 0; i < new_parent.size(); ++i) {
//             new_parent[i] = Find(i);
//           }
//           std::copy(new_parent.begin(), new_parent.end(), parent.begin());
//         };

//     for (index_t j = 0; j < N; ++j) {
//       const index_t rep_j = Find(j);
//       const index_t n_Uj = neib[rep_j].size();
      
//       for (auto iter = patt.outerIndexPtr()[j*dim]; iter < patt.outerIndexPtr()[j*dim+1]; iter += dim) {
//         const index_t i = patt.innerIndexPtr()[iter]/dim;
//         const index_t rep_i = Find(i);

//         // same class
//         if ( rep_i == rep_j ) continue;

//         const index_t n_Ui = neib[rep_i].size();
//         std::vector<index_t> merged_Uij;
//         std::set_union(neib[rep_i].cbegin(), neib[rep_i].cend(),
//                        neib[rep_j].cbegin(), neib[rep_j].cend(),
//                        std::back_inserter(merged_Uij));
//         const index_t n_Ui_Uj = merged_Uij.size();

//         // merge two sets        
//         if ( n_Ui_Uj*n_Ui_Uj <= lambda*(n_Ui*n_Ui+n_Uj*n_Uj) ) {
//           Merge(rep_i, rep_j);
//           neib[rep_i] = merged_Uij;
//           neib[rep_j].clear();
//         }
//       }
//     }

//     Compress(parent);
//     std::map<index_t, index_t> p_map;
//     index_t n_super = 0;
//     for (const auto &pa : parent) {
//       if ( p_map.find(pa) == p_map.end() ) {
//         p_map.insert(std::make_pair(pa, n_super++));
//       }
//     }
//     spdlog::info("nearest n_super={}", n_super);
//     sup_ptr.resize(n_super+1);
//     sup_ind.resize(N);
//     sup_parent.resize(N);

//     std::vector<std::vector<index_t>> sup(n_super);
//     for (index_t k = 0; k < N; ++k) {
//       const auto p_id = p_map[parent[k]];
//       sup_parent[k] = p_id;
//       sup[p_id].emplace_back(k);
//     }

//     sup_ptr[0] = 0;
//     for (index_t i = 1; i < sup_ptr.size(); ++i) {
//       sup_ptr[i] = sup_ptr[i-1]+sup[i-1].size();
//       std::copy(sup[i-1].begin(), sup[i-1].end(), &sup_ind[sup_ptr[i-1]]);
//     }
//     ASSERT(N == sup_ptr[sup_ptr.size()-1]);
//   }
  
//   template <class SpMat, class VecI, class Cont>
//   void aggregate(const index_t  dim,
//                  const Cont     &group,
//                  const SpMat    &patt,
//                  const scalar_t lambda, 
//                  VecI         &sup_ptr,
//                  VecI         &sup_ind,
//                  VecI         &sup_parent,
//                  const index_t max_super_size=INT_MAX) const
//   {
//     typedef typename VecI::Scalar INT;

//     const index_t N = patt.rows()/dim;
//     ASSERT(N == npts_);
//     std::vector<unsigned char> vis(N, 0);

//     std::vector<std::vector<INT>> sup;
//     for (index_t g = 0; g < group.size()-1; ++g) {
//       const auto g_begin = group[g], g_end = group[g+1];

//       for (index_t j = g_begin; j < g_end; ++j) { // for each node
//         if ( vis[j] ) { // has been aggregated into a supernode
//           continue;
//         }
        
//         // start a new supernode
//         sup.emplace_back(std::vector<INT>());
//         for (typename SpMat::InnerIterator it(patt, dim*j); it; ++it) {
//           const index_t i = it.row()/dim;
//           if ( i < g_begin ) continue;
//           if ( i >= g_end ) break;
//           if ( !vis[i] && scale_[i] <= lambda*scale_[j] ) {
//             vis[i] = 1;
//             sup.back().emplace_back(i);
//             if ( sup.back().size() >= max_super_size ) {
//               break;
//             }
//           }
//         }
//       }
//     }
//     const index_t sup_size = sup.size();
//     // spdlog::info("# supernodes={}", sup_size);
    
//     // store supernodes indices
//     sup_ptr.resize(sup_size+1);
//     sup_ind.resize(N);
//     sup_ptr[0] = 0;
//     for (index_t i = 1; i < sup_ptr.size(); ++i) {
//       sup_ptr[i] = sup_ptr[i-1]+sup[i-1].size();
//       std::copy(sup[i-1].begin(), sup[i-1].end(), &sup_ind[sup_ptr[i-1]]);
//     }
//     ASSERT(N == sup_ptr[sup_ptr.size()-1]);    

//     // build parent mapping
//     sup_parent.resize(N);
//     for (index_t i = 0; i < sup.size(); ++i) {
//       for (const auto idx : sup[i]) {
//         sup_parent[idx] = i;
//       }
//     }
//   }

//   template <class SpMat, class VecI> 
//   void super_sparsity(const index_t dim,
//                       const SpMat &patt,
//                       const VecI  &sup_parent,
//                       SpMat       &sup_patt)
//   {
//     typedef typename SpMat::Scalar Scalar;
//     typedef typename SpMat::StorageIndex Index;
    
//     const index_t N = patt.rows()/dim;
//     const index_t n_super = sup_parent.maxCoeff()+1;
//     const SpMat &PT = patt.transpose();
//     // spdlog::info("[super sparsity] n_super={}", n_super);

//     // find the union of row indices for each supernode
//     std::vector<std::vector<Index>> idx_set(n_super);
//     for (Index i = 0; i < PT.cols(); ++i) {
//       for (typename SpMat::InnerIterator it(PT, i); it; ++it) {
//         const Index J = it.row()/dim;
//         const Index PARENT_I = sup_parent[J];
//         idx_set[PARENT_I].emplace_back(i);
//       }
//     }

//     // for (auto &vec : idx_set) {
//     //   vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
//     // }
//     #pragma omp parallel for
//     for (Index i = 0; i < idx_set.size(); ++i) {
//       auto &vec = idx_set[i];
//       vec.erase(std::unique(vec.begin(), vec.end()), vec.end());      
//     }

//     const Index n_dof = patt.cols();
//     Eigen::Matrix<Index, -1, 1> L_ptr(n_dof+1);
//     L_ptr[0] = 0;
//     // for (Index j = 0; j < n_dof; ++j) {
//     //   const Index J = j/dim;
//     //   const Index PJ = sup_parent[J];
//     //   const auto it = std::lower_bound(idx_set[PJ].begin(), idx_set[PJ].end(), j);
//     //   L_ptr[j+1] = L_ptr[j]+(idx_set[PJ].end()-it);
//     // }
//     std::vector<typename std::vector<Index>::iterator> offset(n_dof);
//     #pragma omp parallel for
//     for (Index j = 0; j < n_dof; ++j) {
//       const Index J = j/dim;
//       const Index PJ = sup_parent[J];
//       offset[j] = std::lower_bound(idx_set[PJ].begin(), idx_set[PJ].end(), j); 
//     }
//     for (Index j = 0; j < n_dof; ++j) {
//       const Index J = j/dim;
//       const Index PJ = sup_parent[J];            
//       L_ptr[j+1] = L_ptr[j]+(idx_set[PJ].end()-offset[j]);
//     }
    
//     const Index nnz = L_ptr[L_ptr.size()-1];
//     Eigen::Matrix<Index, -1, 1> L_ind(nnz);
//     //    #pragma omp parallel for
//     for (Index j = 0; j < n_dof; ++j) {
//       const Index J = j/dim;
//       const Index PJ = sup_parent[J];
//       // const auto it = std::lower_bound(idx_set[PJ].begin(), idx_set[PJ].end(), j);
//       std::copy(offset[j], idx_set[PJ].end(), L_ind.data()+L_ptr[j]);
//     }

//     Eigen::Matrix<Scalar, -1, 1> L_val(nnz); L_val.setOnes();
//     sup_patt = Eigen::Map<SpMat>(n_dof, n_dof, nnz, L_ptr.data(), L_ind.data(), L_val.data());
//   }  
  
//  private:
//   index_t npts_;
//   RmMatF_float_t pts_;
//   Eigen::Matrix<scalar_t, -1, 1> scale_, new_scale_;

//   std::shared_ptr<kdtree_t> kdt_;

//   // fine-to-coarse!
//   Eigen::PermutationMatrix<-1, -1> P_, Pinv_;

//  private:
//   void brute_force_fps(const kdtree_t *boundary) {
//     const index_t N = pts_.rows();
//     scale_.resize(N);
    
//     std::vector<index_t> seq;
//     std::vector<unsigned char> vis(N, 0);
//     {
//       // select the first point
//       const Eigen::RowVector3f &corner = pts_.colwise().minCoeff();
//       spdlog::info("corner ({0:.8f}, {1:.8f}, {2:.8f})", corner.x(), corner.y(), corner.z());
      
//       index_t first_idx = -1;
//       if ( boundary ) {
//         std::cout << "# boundary is present!" << std::endl;
//         // select the farthest one to the boundary
//         scalar_t dist_to_corner = 0;
//         for (index_t i = 0; i < npts_; ++i) {
//           size_t ret_index;
//           double dist;
//           nanoflann::KNNResultSet<double> resultSet(1);
//           resultSet.init(&ret_index, &dist);
//           boundary->index->findNeighbors(resultSet, &pts_(i, 0), nanoflann::SearchParams());
//           if ( dist > dist_to_corner ) {
//             dist_to_corner = dist;
//             first_idx = i;
//           }
//         }
//       } else {
//         std::cout << "# boundary is NOT present!" << std::endl;
//         // select the closest to the domain corner
//         scalar_t dist_to_corner = 1e10f;
//         for (index_t i = 0; i < N; ++i) {
//           scalar_t dist = (pts_.row(i)-corner).squaredNorm();
//           if ( dist < dist_to_corner ) {
//             dist_to_corner = dist;
//             first_idx = i;
//           }
//         }
//       }

//       seq.emplace_back(first_idx);
//       vis[first_idx] = 1;
//       scale_[first_idx] = 1e10f; // the first one has inf length scale
//     }
    
//     while ( seq.size() < N ) {
//       if ( seq.size()%100 == 0 ) {
//         std::cout << "processed " << seq.size() << " points" << std::endl;
//       }
      
//       #pragma omp parallel for
//       for (index_t i = 0; i < N; ++i) {
//         if ( !vis[i] ) {
//           scalar_t dist_to_cluster = 1e10f;
//           for (index_t j = 0; j < seq.size(); ++j) {
//             scalar_t dist = (pts_.row(i)-pts_.row(seq[j])).squaredNorm();
//             if ( dist < dist_to_cluster ) {
//               dist_to_cluster = dist;
//             }
//           }

//           double dist_to_boundary = 1e20;
//           if ( boundary ) {
//             size_t ret_index;            
//             nanoflann::KNNResultSet<double> resultSet(1);
//             resultSet.init(&ret_index, &dist_to_boundary);
//             boundary->index->findNeighbors(resultSet, &pts_(i, 0), nanoflann::SearchParams());
//           }

//           scale_[i] = sqrtf(std::min(dist_to_cluster, (scalar_t)dist_to_boundary));
//         }
//       }

//       // find the farthest one
//       scalar_t max_dist = 0;
//       index_t max_idx = -1;
//       for (index_t i = 0; i < N; ++i) {
//         if ( !vis[i] ) {
//           if ( scale_[i] > max_dist ) {
//             max_dist = scale_[i];
//             max_idx = i;
//           }
//         }
//       }
//       vis[max_idx] = 1;
//       seq.emplace_back(max_idx);
//     }

//     // the coarsest goes to the last
//     P_.resize(N);
//     for (index_t i = 0; i < N; ++i) {
//       P_.indices()[seq[i]] = N-1-i;
//     }
//     Pinv_ = P_.inverse();
//     scale_ = P_*scale_;
//     std::cout << "first length=" << scale_.head(1) << std::endl;
//     std::cout << "last length=" << scale_.tail(1) << std::endl;
//   }
//   void fast_fps(const kdtree_t *boundary) {
//     ASSERT(pts_.rows() == npts_);
    
//     // select the first point
//     const Eigen::RowVector3f &corner = pts_.colwise().minCoeff();
//     // spdlog::info("corner ({0:.8f}, {1:.8f}, {2:.8f})", corner.x(), corner.y(), corner.z());
      
//     index_t first_idx = -1;
//     if ( boundary ) {
//       // there is a boundary, select the farthest to the boundary
//       scalar_t dist_to_corner = 0;
//       for (index_t i = 0; i < npts_; ++i) {
//         size_t ret_index;
//         double dist;
//         nanoflann::KNNResultSet<double> resultSet(1);
//         resultSet.init(&ret_index, &dist);
//         boundary->index->findNeighbors(resultSet, &pts_(i, 0), nanoflann::SearchParams());
//         if ( dist > dist_to_corner ) {
//           dist_to_corner = dist;
//           first_idx = i;
//         }
//       }
//     } else {
//       // select the closest to the boundary
//       scalar_t dist_to_corner = 1e10f;      
//       for (index_t i = 0; i < npts_; ++i) {
//         scalar_t dist = (pts_.row(i)-corner).squaredNorm();
//         if ( dist < dist_to_corner ) {
//           dist_to_corner = dist;
//           first_idx = i;
//         }
//       }
//     }
//     // spdlog::info("first_idx={}", first_idx);
    
//     std::vector<unsigned int> seq(npts_), rev_seq(npts_);
//     scale_.resize(npts_);
//     seq[0] = first_idx;
//     create_ordering_3d(&seq[0], &rev_seq[0], scale_.data(), npts_, pts_.data(), first_idx, boundary);

//     std::set<unsigned int> seq_set(seq.begin(), seq.end());
//     ASSERT(seq_set.size() == npts_);

//     P_.resize(npts_);
//     for (index_t i = 0; i < npts_; ++i) {
//       P_.indices()[seq[i]] = npts_-1-i;
//     }
//     Pinv_ = P_.inverse();
//     scale_.reverseInPlace();
//   }
// };

  /**
   * @brief Utility class providing mesh-related functions
   */
class utils_guesss {
public:
  /**
   * @brief Compute the centroid of each face in a mesh
   * 
   * @param V Vertex coordinate matrix; each row is an xyz vertex
   * @param F Face index matrix; each row contains three vertex indices of a face
   * @param V_fc Output; each row stores the centroid of a face
   */
  static void compute_face_centers(const RmMatF_t &V, const RmMatI_t &F, RmMatF_t &V_fc) {
    const int num_faces = F.rows();
    V_fc.resize(num_faces, 3);
    
    for (int i = 0; i < num_faces; ++i) {
      // Get the three vertex indices of the current face
      int v1_idx = F(i, 0);
      int v2_idx = F(i, 1);
      int v3_idx = F(i, 2);
      
      // Face centroid = average of the three vertex coordinates
      V_fc.row(i) = (V.row(v1_idx) + V.row(v2_idx) + V.row(v3_idx)) / 3.0;
    }
  }
};

}

#endif
