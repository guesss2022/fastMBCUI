#ifndef IMG2GRID_H
#define IMG2GRID_H

#include <spdlog/spdlog.h>
#include <opencv2/imgproc.hpp>
#include "macro.h"

namespace klchol {

template <typename T>
struct img_grid_converter
{
  T L_, W_, dx_;
  const cv::Mat img_;

  img_grid_converter(const cv::Mat &img, const T max_len)
      : img_(img)
  {
    if ( img.rows > img.cols ) {
      dx_ = max_len/(img.rows-1);
      L_  = max_len;
      W_  = (img.cols-1)*dx_;
    } else {
      dx_ = max_len/(img.cols-1);
      L_  = (img.rows-1)*dx_;
      W_  = max_len;
    }
    spdlog::info("img info: L={0:.3f}, W={1:.3f}, dx={2:.6f}", L_, W_, dx_);
  }

  double dx() const { return dx_; }

  template <class MatrixV>
  void to_grid(MatrixV &V) const
  {
    const int nx = img_.rows, ny = img_.cols;

    // cv::Mat is row-major
    V.resize(nx*ny, 3);
    #pragma omp parallel for
    for (size_t iter = 0; iter < nx*ny; ++iter) {
      const size_t i = iter/ny, j = iter%ny;
      V(iter, 0) = -0.5*L_+j*dx_;
      V(iter, 1) =  0.5*W_-i*dx_;
      V(iter, 2) = 0;
    }
  }

#if 0  
  template <class MatrixV, class MatrixC>
  void extract_edges(MatrixV &V, MatrixC &Cd) const
  {
    const int nx = img_.rows, ny = img_.cols;

    std::vector<T> coord, color;
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        cv::Vec3b pix = img_.at<cv::Vec3b>(i, j);
        if ( pix[0] > 0 || pix[1] > 0 || pix[2] > 0 ) { // not black
          coord.emplace_back(-0.5*L_+j*dx_);
          coord.emplace_back( 0.5*W_-i*dx_);
          coord.emplace_back(0);

          color.emplace_back(static_cast<T>(pix[0])/255.0);
          color.emplace_back(static_cast<T>(pix[1])/255.0);
          color.emplace_back(static_cast<T>(pix[2])/255.0);
        }
      }
    }

    V.resize(coord.size()/3, 3);
    std::copy(coord.begin(), coord.end(), V.data());
    Cd.resize(color.size()/3, 3);
    std::copy(color.begin(), color.end(), Cd.data());
  }
  
  // Vq for predictions, V for training points  
  // Cd colors for training 
  // return [Vq, V] with P as the permutation, Cd for V
  template <class MatrixV, class MatrixC>
  void separate_pixels(MatrixV &V, MatrixC &Cd,
                       MatrixV &Vq,
                       Eigen::PermutationMatrix<-1, -1> &P) const
  {
    const int nx = img_.rows, ny = img_.cols;

    std::vector<T> coord, color, coord_q;
    std::vector<size_t> seq, seq_q;
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        cv::Vec3b pix = img_.at<cv::Vec3b>(i, j);
        const size_t index = i*ny+j; // cv::Mat is row major

        if ( pix[0] > 0 || pix[1] > 0 || pix[2] > 0 ) { // not black
          coord.emplace_back(-0.5*L_+j*dx_);
          coord.emplace_back( 0.5*W_-i*dx_);
          coord.emplace_back(0);

          color.emplace_back(static_cast<T>(pix[0])/255.0);
          color.emplace_back(static_cast<T>(pix[1])/255.0);
          color.emplace_back(static_cast<T>(pix[2])/255.0);

          seq.emplace_back(index);
        } else {
          coord_q.emplace_back(-0.5*L_+j*dx_);
          coord_q.emplace_back( 0.5*W_-i*dx_);
          coord_q.emplace_back(0);

          seq_q.emplace_back(index);
        }
      }
    }

    V.resize(coord.size()/3, 3);
    std::copy(coord.begin(), coord.end(), V.data());
    Cd.resize(color.size()/3, 3);
    std::copy(color.begin(), color.end(), Cd.data());

    Vq.resize(coord_q.size()/3, 3);
    std::copy(coord_q.begin(), coord_q.end(), Vq.data());

    seq_q.insert(seq_q.end(), seq.begin(), seq.end());
    ASSERT(seq_q.size() == nx*ny);

    P.resize(nx*ny);
    for (size_t i = 0; i < seq_q.size(); ++i) {
      P.indices()[seq_q[i]] = i;
    }
  }
#endif

  template <class MatrixV, class MatrixC, class Cont>
  void select_colored_pixels(MatrixV &V,
                             MatrixC &Cd,
                             Cont &sub_index) const
  {
    typedef typename Cont::Scalar integer_t;
    const int nx = img_.rows, ny = img_.cols;

    std::vector<T> coord, color;
    std::vector<integer_t> select;    

    if ( img_.type() == 0 ) { // CV_8U, C1
      for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
          unsigned char pix = img_.at<unsigned char>(i, j);
          const integer_t index = i*ny+j; // cv::Mat is row major

          if ( pix > 0 ) {
            coord.emplace_back(-0.5*L_+j*dx_);
            coord.emplace_back( 0.5*W_-i*dx_);
            coord.emplace_back(0);

            color.emplace_back(static_cast<T>(pix)/255.0);
            
            select.emplace_back(index);
          }
        }
      }

      Cd.resize(color.size(), 1);
      std::copy(color.begin(), color.end(), Cd.data());
    } else if ( img_.type() == 16 ) { // CV_8U, C3
      for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
          cv::Vec3b pix = img_.at<cv::Vec3b>(i, j);
          const integer_t index = i*ny+j; // cv::Mat is row major

          if ( pix[0] > 0 || pix[1] > 0 || pix[2] > 0 ) { // not black
            coord.emplace_back(-0.5*L_+j*dx_);
            coord.emplace_back( 0.5*W_-i*dx_);
            coord.emplace_back(0);

            color.emplace_back(static_cast<T>(pix[0])/255.0);
            color.emplace_back(static_cast<T>(pix[1])/255.0);
            color.emplace_back(static_cast<T>(pix[2])/255.0);

            select.emplace_back(index);
          }
        }
      }

      Cd.resize(color.size()/3, 3);
      std::copy(color.begin(), color.end(), Cd.data());      
    } else {
      bool img_type_implemented = false;
      ASSERT(img_type_implemented);
    }

    V.resize(coord.size()/3, 3);
    std::copy(coord.begin(), coord.end(), V.data());
    sub_index.resize(select.size());
    std::copy(select.begin(), select.end(), &sub_index[0]);
  }

  template <class MatrixV, class Cont>
  void select_black_pixels(MatrixV &V,
                           Cont &sub_index) const
  {
    typedef typename Cont::Scalar integer_t;
    const int nx = img_.rows, ny = img_.cols;

    std::vector<T> coord, color;
    std::vector<integer_t> select;    
    if ( img_.type() == 0 ) { // CV_8U, C1
      for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
          unsigned char pix = img_.at<unsigned char>(i, j);
          const integer_t index = i*ny+j; // cv::Mat is row major

          if ( pix < 128 ) {
            coord.emplace_back(-0.5*L_+j*dx_);
            coord.emplace_back( 0.5*W_-i*dx_);
            coord.emplace_back(0);
            
            select.emplace_back(index);
          }
        }
      }      
    } else if ( img_.type() == 16 ) { // CV_8U, C3
      for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
          cv::Vec3b pix = img_.at<cv::Vec3b>(i, j);
          const integer_t index = i*ny+j; // cv::Mat is row major

          if ( pix[0] == 0 && pix[1] == 0 && pix[2] == 0 ) {
            coord.emplace_back(-0.5*L_+j*dx_);
            coord.emplace_back( 0.5*W_-i*dx_);
            coord.emplace_back(0);
            
            select.emplace_back(index);
          }
        }
      }
    } else {
      bool img_type_implemented = false;
      ASSERT(img_type_implemented);
    }

    V.resize(coord.size()/3, 3);
    std::copy(coord.begin(), coord.end(), V.data());
    sub_index.resize(select.size());
    std::copy(select.begin(), select.end(), &sub_index[0]);
  }  
};

}

#endif
