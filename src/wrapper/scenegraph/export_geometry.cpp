/* -*-c++-*-
 *  ----------------------------------------------------------------------------
 *
 *       GeomPy: Python wrapper for the Plant Graphic Library
 *
 *       Copyright 1995-2003 UMR AMAP 
 *
 *       File author(s): C. Pradal (christophe.pradal@cirad.fr)
 *
 *       $Source$
 *       $Id$
 *
 *       Forum for AMAPmod developers    : amldevlp@cirad.fr
 *
 *  ----------------------------------------------------------------------------
 *
 *                      GNU General Public Licence
 *
 *       This program is free software; you can redistribute it and/or
 *       modify it under the terms of the GNU General Public License as
 *       published by the Free Software Foundation; either version 2 of
 *       the License, or (at your option) any later version.
 *
 *       This program is distributed in the hope that it will be useful,
 *       but WITHOUT ANY WARRANTY; without even the implied warranty of
 *       MERCHANTABILITY or FITNESS For A PARTICULAR PURPOSE. See the
 *       GNU General Public License for more details.
 *
 *       You should have received a copy of the GNU General Public
 *       License along with this program; see the file COPYING. If not,
 *       write to the Free Software Foundation, Inc., 59
 *       Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  ----------------------------------------------------------------------------
 */

#include "geometry.h"
#include "transformed.h"
#include "primitive.h"
#include "group.h"
#include "text.h"

#include <boost/python.hpp>

#include <scne_sceneobject.h>
#include <geom_geometry.h>

#include "macro_refptr.h"

GEOM_USING_NAMESPACE
TOOLS_USING_NAMESPACE
using namespace boost::python;
using namespace std;

DEF_POINTEE(Geometry)

void class_Geometry()
{

   class_< Geometry,GeometryPtr, bases< SceneObject >, boost::noncopyable > 
      ("Geometry",no_init)
     .def("isACurve",&Geometry::isACurve)
     .def("isASurface",&Geometry::isASurface)
     .def("isAVolume",&Geometry::isAVolume)
     .def("isExplicit",&Geometry::isExplicit)
;
   
   implicitly_convertible<GeometryPtr, SceneObjectPtr >();
   
}


void module_geometry()
{
  class_Geometry();
  class_Transformed();
  class_Primitive();
  class_Group();
  class_Text();
}