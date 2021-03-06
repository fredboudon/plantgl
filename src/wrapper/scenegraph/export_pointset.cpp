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



#include <plantgl/scenegraph/geometry/pointset.h>
#include <plantgl/scenegraph/container/pointarray.h>
#include <plantgl/scenegraph/transformation/transformed.h>
#include <plantgl/scenegraph/geometry/mesh.h>
#include <plantgl/scenegraph/scene/scene.h>
#include <plantgl/scenegraph/appearance/material.h>

#include <plantgl/python/export_refcountptr.h>
#include <plantgl/python/export_property.h>
#include "export_sceneobject.h"
#include <plantgl/python/extract_list.h>

#include <boost/python/make_constructor.hpp>
#include <string>
#include <sstream>

using namespace boost::python;
#define bp boost::python

PGL_USING_NAMESPACE

DEF_POINTEE( PointSet )

object ps_findclosest(PointSet * lm, Vector3 point)
{
    uint_t u;
    Vector3 res = lm->findClosest(point,&u);
    return make_tuple(res,u);
}

std::string gps_repr( PointSet* p )
{
  std::stringstream ss;
  ss << "PointSet(" << extract<std::string>(boost::python::str(boost::python::object(p->getPointList())))();
  if (p->getColorList())
    ss << "," << extract<std::string>(boost::python::str(boost::python::object(p->getColorList())))() ;
  ss << ")";
  return ss.str();
}

ScenePtr convertPointSetToShapes(PointSet * ps, Vector3 translation = Vector3(0,0,0)) {
    Point3ArrayPtr points = ps->getPointList();
    Color4ArrayPtr colors = ps->getColorList();
    Color4Array::const_iterator itColor;
    if(colors) itColor = colors->begin();
    ScenePtr result(new Scene());
    size_t id = 0;
    AppearancePtr mat(new Material("DefaultPointSetMat"));
    for(Point3Array::const_iterator itPoint = points->begin(); itPoint != points->end(); ++itPoint,++id){
        result->add(Shape3DPtr(new Shape(GeometryPtr(new PointSet(Point3ArrayPtr(new Point3Array(1,(*itPoint)+translation)),
                                                                  Color4ArrayPtr(colors?new Color4Array(1,*itColor):NULL))),
                                          mat,id)));
        if(colors) ++itColor;
    }
    return result;
}

void export_PointSet()
{
  class_< PointSet, PointSetPtr, bases<ExplicitModel>, boost::noncopyable>( "PointSet",
      "PointSet describes an explicit set of points",
      init<Point3ArrayPtr, optional<Color4ArrayPtr, uchar_t> >("PointSet(Point3Array pointList, Color4Array colorList = None)",
      (bp::arg("pointList"),bp::arg("colorList")=Color4ArrayPtr(),bp::arg("width") = PointSet::DEFAULT_WIDTH)))
    .DEF_PGLBASE(PointSet)
    .def( "__repr__", gps_repr )
    .def( "transform", &PointSet::transform )
    .DEC_BT_NR_PROPERTY_WDV(width,PointSet,Width,uchar_t,DEFAULT_WIDTH)
    .def("convertToShapes",&convertPointSetToShapes,(bp::arg("translation")=Vector3(0,0,0)))
    .def("findClosest",&ps_findclosest,"Find closest point in the PointSet from arg",args("point"))
    ;
  implicitly_convertible<PointSetPtr, ExplicitModelPtr>();
}

DEF_POINTEE( PointSet2D )

object ps2_findclosest(PointSet2D * lm, Vector2 point)
{
    uint_t u;
    Vector2 res = lm->findClosest(point,&u);
    return make_tuple(res,u);
}

std::string gps2d_repr( PointSet2D* p )
{
  std::stringstream ss;
  ss << "PointSet2D(" << extract<std::string>(boost::python::str(boost::python::object(p->getPointList())))() << ")";
  return ss.str();
}


void export_PointSet2D()
{
  class_< PointSet2D, PointSet2DPtr, bases<PlanarModel>, boost::noncopyable>(
      "PointSet2D", "PointSet2D describes an explicit set of 2D points. See PointSet.",
      init<Point2ArrayPtr,optional<uchar_t> >("PointSet2D(pointList[,width])",(bp::arg("pointList"),bp::arg("width") = PointSet::DEFAULT_WIDTH)) )
    .DEF_PGLBASE(PointSet2D)
    .def( "__repr__", gps2d_repr )
    .DEC_PTR_PROPERTY(pointList,PointSet2D,PointList,Point2ArrayPtr)
    .DEC_BT_NR_PROPERTY(width,PointSet2D,Width,uchar_t)
    DEC_DEFAULTVALTEST(PointSet2D,Width)
    .def("findClosest",&ps2_findclosest,"Find closest point in the PointSet from arg",args("point"))
    ;
  implicitly_convertible<PointSet2DPtr, PlanarModelPtr>();
}

