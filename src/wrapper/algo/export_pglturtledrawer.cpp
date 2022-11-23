/* -*-c++-*-
 *  ----------------------------------------------------------------------------
 *
 *       PlantGL: The Plant Graphic Library
 *
 *       Copyright CIRAD/INRIA/INRA
 *
 *       File author(s): F. Boudon (frederic.boudon@cirad.fr) et al. 
 *
 *  ----------------------------------------------------------------------------
 *
 *   This software is governed by the CeCILL-C license under French law and
 *   abiding by the rules of distribution of free software.  You can  use, 
 *   modify and/ or redistribute the software under the terms of the CeCILL-C
 *   license as circulated by CEA, CNRS and INRIA at the following URL
 *   "http://www.cecill.info". 
 *
 *   As a counterpart to the access to the source code and  rights to copy,
 *   modify and redistribute granted by the license, users are provided only
 *   with a limited warranty  and the software's author,  the holder of the
 *   economic rights,  and the successive licensors  have only  limited
 *   liability. 
 *       
 *   In this respect, the user's attention is drawn to the risks associated
 *   with loading,  using,  modifying and/or developing or reproducing the
 *   software by the user in light of its specific status of free software,
 *   that may mean  that it is complicated to manipulate,  and  that  also
 *   therefore means  that it is reserved for developers  and  experienced
 *   professionals having in-depth computer knowledge. Users are therefore
 *   encouraged to load and test the software's suitability as regards their
 *   requirements in conditions enabling the security of their systems and/or 
 *   data to be ensured and,  more generally, to use and operate it in the 
 *   same conditions as regards security. 
 *
 *   The fact that you are presently reading this means that you have had
 *   knowledge of the CeCILL-C license and that you accept its terms.
 *
 *  ----------------------------------------------------------------------------
 */



#include <plantgl/algo/modelling/pglturtledrawer.h>
#include <plantgl/python/export_property.h>
#include <plantgl/python/export_list.h>
#include <plantgl/python/extract_list.h>

#include <boost/python.hpp>
using namespace boost::python;
#define bp boost::python
PGL_USING_NAMESPACE


void export_PglTurtleDrawer()
{
  class_< PglTurtleDrawer , boost::noncopyable, bases<TurtleDrawer> >("PglTurtleDrawer", init<>("PglTurtleDrawer() -> Create PglTurtleDrawer"))

    .def("cylinder",
         (void (PglTurtleDrawer::*) (const id_pair, const FrameInfo&, AppearancePtr, real_t, real_t, uint_t)) &PglTurtleDrawer::cylinder,
         return_self<>())
    .def("frustum",
         (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, real_t, real_t, real_t, uint_t)) &PglTurtleDrawer::frustum,
         return_self<>())
    .def("generalizedCylinder",
         (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, const Point3ArrayPtr&,
                 const std::vector<Vector3>&, const std::vector<real_t>&, const Curve2DPtr&, bool, uint_t)) &PglTurtleDrawer::generalizedCylinder,
         return_self<>())
    .def("sphere",
         (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, real_t, uint_t)) &PglTurtleDrawer::sphere,
         return_self<>() )
    .def("circle",
         (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, real_t, uint_t)) &PglTurtleDrawer::circle,
         return_self<>() )
    .def("box",
         (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, real_t, real_t, real_t)) &PglTurtleDrawer::box,
         return_self<>())
    .def("quad",
         (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, real_t, real_t, real_t)) &PglTurtleDrawer::quad,
         return_self<>())
    .def("polygon",
       (void (PglTurtleDrawer::*) (const id_pair, const FrameInfo&, AppearancePtr, const Point3ArrayPtr&, bool)) &PglTurtleDrawer::polygon,
       return_self<>())
    .def("arrow",
       (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, real_t, real_t, real_t, real_t, uint_t)) &PglTurtleDrawer::arrow,
       return_self<>())
    .def("arrow_color",
       (void (PglTurtleDrawer::*) (const id_pair, const FrameInfo&, real_t, real_t, real_t, real_t, real_t, real_t, uint_t)) &PglTurtleDrawer::arrow,
       return_self<>())
    .def("frame",
       (void (PglTurtleDrawer::*) (const id_pair, const FrameInfo&, real_t, real_t, real_t, real_t, real_t, real_t, uint_t)) &PglTurtleDrawer::frame,
       return_self<>() )
    .def("label",
       (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, const std::string&, bool, int)) &PglTurtleDrawer::label,
       return_self<>() )
    .def("customGeometry",
       (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, const GeometryPtr, real_t)) &PglTurtleDrawer::customGeometry,
       return_self<>())
    .def("smallSweep",
       (void (PglTurtleDrawer::*) (const id_pair, AppearancePtr, const FrameInfo&, const real_t length, const real_t,
               const real_t, const Curve2DPtr&, bool, uint_t)) &PglTurtleDrawer::smallSweep,
       return_self<>())
    ;
}
