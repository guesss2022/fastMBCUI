#include "io.h"
#include <numeric>
#include "macro.h"
#include "vtk.h"

using namespace std;

namespace mschol {

// int tri_mesh_write_to_vtk(const char *path, const matd_t &nods,
//                           const mati_t &tris, const matd_t *mtr,
//                           const char *type) {
//   ASSERT(tris.rows() == 3);

//   ofstream ofs(path);
//   if ( ofs.fail() )
//     return __LINE__;

//   ofs << setprecision(15);

//   if ( nods.rows() == 2 ) {
//     matd_t tmp_nods = matd_t::Zero(3, nods.cols());
//     tmp_nods.row(0) = nods.row(0);
//     tmp_nods.row(1) = nods.row(1);
//     tri2vtk(ofs, tmp_nods.data(), tmp_nods.cols(), tris.data(), tris.cols());
//   } else if ( nods.rows() == 3) {
//     tri2vtk(ofs, nods.data(), nods.size()/3, tris.data(), tris.size()/3);
//   }

//   if ( mtr != nullptr ) {
//     for (int i = 0; i < mtr->rows(); ++i) {
//       const string mtr_name = "theta_"+to_string(i);
//       const matd_t curr_mtr = mtr->row(i);
//       if ( i == 0 )
//         ofs << type << "_DATA " << curr_mtr.size() << "\n";
//       vtk_data(ofs, curr_mtr.data(), curr_mtr.size(), mtr_name.c_str(), mtr_name.c_str());
//     }
//   }
//   ofs.close();
//   return 0;
// }

// int tet_mesh_write_to_vtk(const char *path, const matd_t &nods, const mati_t &tets,
//                           const matd_t *mtr, const char *type) {
//   ASSERT(tets.rows() == 4);
  
//   ofstream ofs(path);
//   if ( ofs.fail() )
//     return __LINE__;
  
//   ofs << setprecision(15);
//   tet2vtk(ofs, nods.data(), nods.size()/3, tets.data(), tets.cols());

//   if ( mtr != nullptr ) {
//     for (int i = 0; i < mtr->rows(); ++i) {
//       const string mtr_name = "theta_"+to_string(i);
//       const matd_t curr_mtr = mtr->row(i);
//       if ( i == 0 )
//         ofs << type << "_DATA " << curr_mtr.size() << "\n";
//       vtk_data(ofs, curr_mtr.data(), curr_mtr.size(), mtr_name.c_str(), mtr_name.c_str());
//     }
//   }
//   ofs.close();
//   return 0;
// }

// int point_write_to_vtk(const char *path, const matd_t &nods) {
//   ofstream ofs(path);
//   if ( ofs.fail() )
//     return __LINE__;

//   matd_t nods_to_write = matd_t::Zero(3, nods.cols());
//   if ( nods.rows() == 2 ) {
//     nods_to_write.topRows(2) = nods;
//   } else if ( nods.rows() == 3 ) {
//     nods_to_write = nods;
//   }

//   std::vector<size_t> cell(nods.cols());
//   std::iota(cell.begin(), cell.end(), 0);
//   point2vtk(ofs, nods_to_write.data(), nods_to_write.cols(), &cell[0], cell.size());
//   ofs.close();
        
//   return 0;
// }

}
