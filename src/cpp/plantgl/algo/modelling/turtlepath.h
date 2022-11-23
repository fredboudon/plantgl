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

#ifndef __PGL_TURTLE_PATH_H__
#define __PGL_TURTLE_PATH_H__

#include "../algo_config.h"
#include <plantgl/math/util_vector.h>
#include <plantgl/math/util_matrix.h>
#include <plantgl/scenegraph/appearance/color.h>
#include <plantgl/scenegraph/geometry/curve.h>
#include <plantgl/scenegraph/geometry/lineicmodel.h>
#include <plantgl/scenegraph/function/function.h>
#include <plantgl/scenegraph/appearance/appearance.h>
#include <vector>

PGL_BEGIN_NAMESPACE

class TurtlePath;
typedef RCPtr<TurtlePath> TurtlePathPtr;

/// Class that contains a path parameter that should be followed by the turtle
class ALGO_API TurtlePath : public RefCountObject{
public:
    TurtlePath(real_t totalLength, real_t actualLength, QuantisedFunctionPtr arclengthParam = QuantisedFunctionPtr()) : __totalLength(totalLength), __actualLength(actualLength), __scale(totalLength/actualLength), __arclengthParam(arclengthParam), __actualT(0)  { }
    virtual ~TurtlePath();

    virtual bool is2D() const { return true; }

    virtual TurtlePathPtr copy() const = 0;

    virtual void setPosition(real_t t)  = 0;

    real_t __totalLength;
    real_t __actualLength;
    real_t __scale;
    QuantisedFunctionPtr __arclengthParam;
    real_t __actualT;
};

/// Class that contains a 2D path parameter that should be followed by the turtle
class ALGO_API Turtle2DPath : public TurtlePath {
public:
    Turtle2DPath(Curve2DPtr curve, real_t totalLength, real_t actualLength, bool orientation = false, bool ccw = false, QuantisedFunctionPtr arclengthParam = QuantisedFunctionPtr());

    virtual TurtlePathPtr copy() const;
    virtual void setPosition(real_t t) ;

    // Path to follow
    Curve2DPtr __path;
    // Tell whether path is oriented with Y as first heading or X
    bool __orientation;
    // Tell whether the resulting structure is in CCW
    bool __ccw;

    // Position on the curve
    Vector2 __lastPosition;
    // Last direction on the curve
    Vector2 __lastHeading;
};

/// Class that contains a 2D path parameter that should be followed by the turtle
class ALGO_API Turtle3DPath : public TurtlePath {
public:
    Turtle3DPath(LineicModelPtr curve, real_t totalLength, real_t actualLength, QuantisedFunctionPtr arclengthParam = QuantisedFunctionPtr());

    virtual TurtlePathPtr copy() const;
    virtual void setPosition(real_t t) ;

    virtual bool is2D() const { return false; }

    // 3D Path to follow
    LineicModelPtr __path;

    // Position on the curve
    Vector3 __lastPosition;

    // Reference frame on the curve
    Vector3 __lastHeading;
    Vector3 __lastUp;
    Vector3 __lastLeft;

};

struct PathInfo {
    real_t length;
    QuantisedFunctionPtr arclengthParam;
};

typedef pgl_hash_map<size_t,PathInfo> PathInfoMap;


/* ----------------------------------------------------------------------- */

PGL_END_NAMESPACE

/* ----------------------------------------------------------------------- */

#endif
