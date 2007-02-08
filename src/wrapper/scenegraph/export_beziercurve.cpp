#include "beziercurve.h"

#include <geom_beziercurve.h>
#include <geom_pointarray.h>
#include <geom_mesh.h>

#include <boost/python.hpp>
#include <string>
#include <sstream>

#include "macro_refptr.h"
#include "macro_property.h"

using namespace boost::python;

GEOM_USING_NAMESPACE
TOOLS_USING_NAMESPACE

DEF_POINTEE( BezierCurve )
DEF_POINTEE( BezierCurve2D )

SETGET(BezierCurve,Stride,uint32_t);
SETGET(BezierCurve2D,Stride,uint32_t);
SETGET(BezierCurve,CtrlPointList,Point4ArrayPtr);
SETGET(BezierCurve2D,CtrlPointList,Point3ArrayPtr);

std::string gbc_repr( BezierCurve* p )
{
  std::stringstream ss;
  Point4ArrayPtr ctrl= p->getCtrlPoints();
  uint32_t stride= p->getStride();
  uint32_t n= ctrl->getSize();
  if( n == 0 )
    {
      ss << "BezierCurve(Point4Array([])," << stride << ")";
      return ss.str();
    }

  Vector4 v = ctrl->getAt( 0 );
  ss << "BezierCurve(Point4Array([Vector4(" << v.x() << ", " << v.y()
     << ", " << v.z() << ", " << v.w() << ")";
  size_t i;
  for( i = 1 ; i < n ; ++i )
    {
      v = ctrl->getAt( i );
      ss << ", Vector4(" << v.x() << ", " << v.y() << ", " << v.z() << ", " << v.w() << ")";
    }
  ss << "])," << stride <<")";
  return ss.str();
}

void class_BezierCurve()
{
  class_<BezierCurve, BezierCurvePtr, bases<ParametricModel, LineicModel>, boost::noncopyable>
    ( "BezierCurve", init<Point4ArrayPtr, optional< uint32_t > >() )
    .def( "copy", &BezierCurve::copy )
    .def( "getPointAt", &BezierCurve::getPointAt )
    .def( "__repr__", gbc_repr )
    .DEC_SETGET_WD(stride,BezierCurve,Stride,uint32_t)
    .DEC_SETGET(ctrlPointList,BezierCurve,CtrlPointList,Point4ArrayPtr)
    ;

  implicitly_convertible<BezierCurvePtr, ParametricModelPtr>();
  implicitly_convertible<BezierCurvePtr, LineicModelPtr>();
}

std::string gbc2_repr( BezierCurve2D* p )
{
  std::stringstream ss;
  Point3ArrayPtr ctrl= p->getCtrlPoints();
  uint32_t stride= p->getStride();
  uint32_t n= ctrl->getSize();
  if( n == 0 )
    {
      ss << "BezierCurve2D(Point3Array([])," << stride << ")";
      return ss.str();
    }

  Vector3 v = ctrl->getAt( 0 );
  ss << "BezierCurve2D(Point3Array([Vector3(" << v.x() << ", " << v.y()
     << ", " << v.z() <<  ")";
  size_t i;
  for( i = 1 ; i < n ; ++i )
    {
      v = ctrl->getAt( i );
      ss << ", Vector3(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
    }
  ss << "])," << stride <<")";
  return ss.str();
}

void class_BezierCurve2D()
{
   class_<BezierCurve2D, BezierCurve2DPtr, bases<Curve2D>, boost::noncopyable>
    ( "BezierCurve2D", init<Point3ArrayPtr, optional< uint32_t > >() )
    .def( "copy", &BezierCurve2D::copy )
    .def( "getPointAt", &BezierCurve2D::getPointAt )
    .def( "__repr__", gbc2_repr )
    .DEC_SETGET_WD(stride,BezierCurve2D,Stride,uint32_t)
    .DEC_SETGET(ctrlPointList,BezierCurve2D,CtrlPointList,Point3ArrayPtr)
    ;

  implicitly_convertible<BezierCurve2DPtr, Curve2DPtr>();
}

