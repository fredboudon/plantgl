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



#include <plantgl/math/util_math.h>
#include "asymmetrichull.h"
#include <plantgl/scenegraph/core/pgl_messages.h>

PGL_USING_NAMESPACE

/* ----------------------------------------------------------------------- */

const real_t AsymmetricHull::DEFAULT_NEG_X_HEIGHT(0);
const real_t AsymmetricHull::DEFAULT_NEG_X_RADIUS(.5);
const real_t AsymmetricHull::DEFAULT_NEG_Y_HEIGHT(0);
const real_t AsymmetricHull::DEFAULT_NEG_Y_RADIUS(.5);
const real_t AsymmetricHull::DEFAULT_POS_X_HEIGHT(0);
const real_t AsymmetricHull::DEFAULT_POS_X_RADIUS(.5);
const real_t AsymmetricHull::DEFAULT_POS_Y_HEIGHT(0);
const real_t AsymmetricHull::DEFAULT_POS_Y_RADIUS(.5);
const Vector3 AsymmetricHull::DEFAULT_BOTTOM(0,0,-.5);
const Vector3 AsymmetricHull::DEFAULT_TOP(0,0,.5);
const real_t AsymmetricHull::DEFAULT_BOTTOM_SHAPE(2);
const real_t AsymmetricHull::DEFAULT_TOP_SHAPE(2);
const uchar_t AsymmetricHull::DEFAULT_SLICES(4);
const uchar_t AsymmetricHull::DEFAULT_STACKS(4);


/* ----------------------------------------------------------------------- */


AsymmetricHull::Builder::Builder( ) :
  Hull::Builder(),
  NegXHeight(0),
  NegXRadius(0),
  NegYHeight(0),
  NegYRadius(0),
  PosXHeight(0),
  PosXRadius(0),
  PosYHeight(0),
  PosYRadius(0),
  Bottom(0),
  Top(0),
  BottomShape(0),
  TopShape(0),
  Slices(0),
  Stacks(0) {
}


AsymmetricHull::Builder::~Builder( ) {
  // nothing to do
}


SceneObjectPtr AsymmetricHull::Builder::build( ) const {
  if (isValid())
    return SceneObjectPtr
      (new AsymmetricHull
       (
    NegXRadius ? *NegXRadius : DEFAULT_NEG_X_RADIUS,
    PosXRadius ? *PosXRadius : DEFAULT_POS_X_RADIUS,
    NegYRadius ? *NegYRadius : DEFAULT_NEG_Y_RADIUS,
    PosYRadius ? *PosYRadius : DEFAULT_POS_Y_RADIUS,
    NegXHeight ? *NegXHeight : DEFAULT_NEG_X_HEIGHT,
    PosXHeight ? *PosXHeight : DEFAULT_POS_X_HEIGHT,
    NegYHeight ? *NegYHeight : DEFAULT_NEG_Y_HEIGHT,
    PosYHeight ? *PosYHeight : DEFAULT_POS_Y_HEIGHT,
    Bottom ? *Bottom : DEFAULT_BOTTOM,
    Top ? *Top : DEFAULT_TOP,
    BottomShape ? *BottomShape : DEFAULT_BOTTOM_SHAPE,
    TopShape ? *TopShape : DEFAULT_TOP_SHAPE,
    Slices ? *Slices : DEFAULT_SLICES,
    Stacks ? *Stacks : DEFAULT_STACKS));
  // Returns null as self is not valid.
  return SceneObjectPtr();
}


void AsymmetricHull::Builder::destroy( ) {
  if (NegXHeight) delete NegXHeight;
  if (NegXRadius) delete NegXRadius;
  if (NegYHeight) delete NegYHeight;
  if (NegYRadius) delete NegYRadius;
  if (PosXHeight) delete PosXHeight;
  if (PosXRadius) delete PosXRadius;
  if (PosYHeight) delete PosYHeight;
  if (PosYRadius) delete PosYRadius;
  if (Bottom) delete Bottom;
  if (Top) delete Top;
  if (BottomShape) delete BottomShape;
  if (TopShape) delete TopShape;
  if (Slices) delete Slices;
  if (Stacks) delete Stacks;
}


bool AsymmetricHull::Builder::isValid( ) const {

  // Bottom field
  if (Bottom &&
      ((! Bottom->isValid()) /*||
       (Bottom->z() > (Top ? Top->z() : DEFAULT_TOP.z())) ||
       (Bottom->x() < -(NegXRadius ? *NegXRadius : DEFAULT_NEG_X_RADIUS)) ||
       (Bottom->x() > (PosXRadius ? *PosXRadius : DEFAULT_POS_X_RADIUS)) ||
       (Bottom->y() < -(NegYRadius ? *NegYRadius : DEFAULT_NEG_Y_RADIUS)) ||
       (Bottom->y() > (PosYRadius ? *PosYRadius : DEFAULT_POS_Y_RADIUS))*/)) {
    pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","Bottom","Depend on Top, NegXRadius, PosXRadius, NegYRadius and PosYRadius values.");
    return false;
  };

  // Top field
  if (Top &&
      ((! Top->isValid()) /*||
       (Top->z() < (Bottom ? Bottom->z() : DEFAULT_BOTTOM.z())) ||
       (Top->x() < -(NegXRadius ? *NegXRadius : DEFAULT_NEG_X_RADIUS)) ||
       (Top->x() > (PosXRadius ? *PosXRadius : DEFAULT_POS_X_RADIUS)) ||
       (Top->y() < -(NegYRadius ? *NegYRadius : DEFAULT_NEG_Y_RADIUS)) ||
       (Top->y() > (PosYRadius ? *PosYRadius : DEFAULT_POS_Y_RADIUS))*/)) {
    pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","Top","Depend on Bottom, NegXRadius, PosXRadius, NegYRadius and PosYRadius values.");
    return false;
  };

  // NegXRadius
  if (NegXRadius &&
      ((! pglfinite(*NegXRadius)) /*||
       (*NegXRadius < 0)*/)) {
      pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","NegXRadius","Must be Finite and Positive.");
      return false;
    };

  // PosXRadius
  if (PosXRadius &&
      ((! pglfinite(*PosXRadius)) /*||
       (*PosXRadius < 0)*/)) {
      pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","PosXRadius","Must be Finite and Positive.");
      return false;
    };

  // NegYRadius
  if (NegYRadius &&
      ((! pglfinite(*NegYRadius)) /*||
       (*NegYRadius < 0)*/)) {
      pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","NegYRadius","Must be Finite and Positive.");
      return false;
    };

  // PosYRadius
  if (PosYRadius &&
      ((! pglfinite(*PosYRadius)) /*||
       (*PosYRadius < 0)*/)) {
      pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","PosYRadius","Must be Finite and Positive.");
      return false;
    };

  // BottomShape
  if (BottomShape &&
      ( !(*BottomShape > GEOM_EPSILON))) {
      pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","BottomShape","Must be not null.");
      return false;
    };

  // TopShape
  if (TopShape &&
      ( !(*TopShape > GEOM_EPSILON))) {
      pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","TopShape","Must be not null.");
      return false;
    };

  // Slices field
  if (Slices &&
      (*Slices < 1)) {
    pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","Slices","Must be greater than 0.");
    return false;
  };

  // Stacks field
  if (Stacks &&
      (*Stacks < 1)) {
    pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Asymmetric Hull","Stacks","Must be greater than 0.");
    return false;
  };

  return true;
};


/* ----------------------------------------------------------------------- */


AsymmetricHull::AsymmetricHull( const real_t& negXRadius,
                const real_t& posXRadius,
                const real_t& negYRadius,
                const real_t& posYRadius,
                const real_t& negXHeight,
                const real_t& posXHeight,
                const real_t& negYHeight,
                const real_t& posYHeight,
                const Vector3& bottom,
                const Vector3& top,
                const real_t& bottomShape,
                const real_t& topShape,
                uchar_t slices,
                uchar_t stacks ) :
  Hull(),
  __negXHeight(negXHeight),
  __negXRadius(negXRadius),
  __negYHeight(negYHeight),
  __negYRadius(negYRadius),
  __posXHeight(posXHeight),
  __posXRadius(posXRadius),
  __posYHeight(posYHeight),
  __posYRadius(posYRadius),
  __top(top),
  __bottom(bottom),
  __topShape(topShape),
  __bottomShape(bottomShape),
  __slices(slices),
  __stacks(stacks) {
  GEOM_ASSERT(isValid());
}

AsymmetricHull::~AsymmetricHull( ) {
#ifdef GEOM_DEBUG
    cerr <<"AsymmetricHull " <<  __name << " destroyed" << endl;
#endif
}

bool AsymmetricHull::isValid( ) const {
  Builder _builder;
  _builder.NegXHeight = const_cast<real_t *>(&__negXHeight);
  _builder.NegXRadius = const_cast<real_t *>(&__negXRadius);
  _builder.NegYHeight = const_cast<real_t *>(&__negYHeight);
  _builder.NegYRadius = const_cast<real_t *>(&__negYRadius);
  _builder.PosXHeight = const_cast<real_t *>(&__posXHeight);
  _builder.PosXRadius = const_cast<real_t *>(&__posXRadius);
  _builder.PosYHeight = const_cast<real_t *>(&__posYHeight);
  _builder.PosYRadius = const_cast<real_t *>(&__posYRadius);
  _builder.Bottom = const_cast<Vector3 *>(&__bottom);
  _builder.Top = const_cast<Vector3 *>(&__top);
  _builder.BottomShape = const_cast<real_t *>(&__bottomShape);
  _builder.TopShape = const_cast<real_t *>(&__topShape);
  _builder.Slices = const_cast<uchar_t *>(&__slices);
  _builder.Stacks = const_cast<uchar_t *>(&__stacks);
  return _builder.isValid();
}

SceneObjectPtr AsymmetricHull::copy(DeepCopier& copier) const
{
  return SceneObjectPtr(new AsymmetricHull(*this));
}

/* ----------------------------------------------------------------------- */

const Vector3&
AsymmetricHull::getBottom( ) const {
  return __bottom;
}

Vector3&
AsymmetricHull::getBottom( ){
  return __bottom;
}

const real_t&
AsymmetricHull::getBottomShape( ) const {
  return __bottomShape;
}

real_t&
AsymmetricHull::getBottomShape( ){
  return __bottomShape;
}

const real_t&
AsymmetricHull::getNegXHeight( ) const {
  return __negXHeight;
}

real_t&
AsymmetricHull::getNegXHeight( ) {
  return __negXHeight;
}

const real_t&
AsymmetricHull::getNegXRadius( ) const {
  return __negXRadius;
}

real_t&
AsymmetricHull::getNegXRadius( ) {
  return __negXRadius;
}

Vector3
AsymmetricHull::getNegXPoint( ) const {
  return Vector3( - __negXRadius,0,__negXHeight);
}

const real_t&
AsymmetricHull::getNegYHeight( ) const {
  return __negYHeight;
}

real_t&
AsymmetricHull::getNegYHeight( ) {
  return __negYHeight;
}

const real_t&
AsymmetricHull::getNegYRadius( ) const {
  return __negYRadius;
}

real_t&
AsymmetricHull::getNegYRadius( ) {
  return __negYRadius;
}

Vector3
AsymmetricHull::getNegYPoint( ) const {
  return Vector3(0,- __negYRadius,__negYHeight);
}

const real_t&
AsymmetricHull::getPosXHeight( ) const {
  return __posXHeight;
}

real_t&
AsymmetricHull::getPosXHeight( ){
  return __posXHeight;
}

const real_t&
AsymmetricHull::getPosXRadius( ) const {
  return __posXRadius;
}

real_t&
AsymmetricHull::getPosXRadius( )  {
  return __posXRadius;
}

Vector3
AsymmetricHull::getPosXPoint( ) const {
  return Vector3(__posXRadius,0,__posXHeight);
}

const real_t&
AsymmetricHull::getPosYHeight( ) const {
  return __posYHeight;
}

real_t&
AsymmetricHull::getPosYHeight( ) {
  return __posYHeight;
}

const real_t&
AsymmetricHull::getPosYRadius( ) const {
  return __posYRadius;
}

real_t&
AsymmetricHull::getPosYRadius( ){
  return __posYRadius;
}

Vector3
AsymmetricHull::getPosYPoint( ) const {
  return Vector3(0,__posYRadius,__posYHeight);
}

const Vector3&
AsymmetricHull::getTop( ) const {
  return __top;
}

Vector3&
AsymmetricHull::getTop( )  {
  return __top;
}

const real_t&
AsymmetricHull::getTopShape( ) const {
  return __topShape;
}

real_t&
AsymmetricHull::getTopShape( ) {
  return __topShape;
}

const uchar_t&
AsymmetricHull::getSlices( ) const {
  return __slices;
}

uchar_t&
AsymmetricHull::getSlices( ) {
  return __slices;
}

const uchar_t&
AsymmetricHull::getStacks( ) const {
  return __stacks;
}

uchar_t&
AsymmetricHull::getStacks( ) {
  return __stacks;
}

bool
AsymmetricHull::isBottomToDefault( ) const {
  return __bottom == DEFAULT_BOTTOM;
}

bool
AsymmetricHull::isBottomShapeToDefault( ) const {
  return fabs(__bottomShape - DEFAULT_BOTTOM_SHAPE) < GEOM_EPSILON;
}

bool
AsymmetricHull::isNegXHeightToDefault( ) const {
  return fabs(__negXHeight - DEFAULT_NEG_X_HEIGHT) < GEOM_EPSILON;
}

bool
AsymmetricHull::isNegXRadiusToDefault( ) const {
  return fabs(__negXRadius - DEFAULT_NEG_X_RADIUS) < GEOM_EPSILON;
}

bool
AsymmetricHull::isNegYHeightToDefault( ) const {
  return fabs(__negYHeight - DEFAULT_NEG_Y_HEIGHT) < GEOM_EPSILON;
}

bool
AsymmetricHull::isNegYRadiusToDefault( ) const {
  return fabs(__negYRadius - DEFAULT_NEG_Y_RADIUS) < GEOM_EPSILON;
}

bool
AsymmetricHull::isPosXHeightToDefault( ) const {
  return fabs(__posXHeight - DEFAULT_POS_X_HEIGHT) < GEOM_EPSILON;
}

bool
AsymmetricHull::isPosXRadiusToDefault( ) const {
  return fabs(__posXRadius - DEFAULT_POS_X_RADIUS) < GEOM_EPSILON;
}

bool
AsymmetricHull::isPosYHeightToDefault( ) const {
  return fabs(__posYHeight - DEFAULT_POS_Y_HEIGHT) < GEOM_EPSILON;
}

bool
AsymmetricHull::isPosYRadiusToDefault( ) const {
  return fabs(__posYRadius - DEFAULT_POS_Y_RADIUS) < GEOM_EPSILON;
}

bool
AsymmetricHull::isTopToDefault( ) const {
  return __top == DEFAULT_TOP;
}

bool
AsymmetricHull::isTopShapeToDefault( ) const {
  return fabs(__topShape - DEFAULT_TOP_SHAPE) < GEOM_EPSILON;
}

bool
AsymmetricHull::isSlicesToDefault( ) const {
  return __slices == DEFAULT_SLICES;
}

bool
AsymmetricHull::isStacksToDefault( ) const {
  return __stacks == DEFAULT_STACKS;
}





/* ----------------------------------------------------------------------- */
