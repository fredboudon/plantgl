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





#include "mattransformed.h"
#include "orthotransformed.h"

#include <plantgl/scenegraph/container/pointarray.h>
#include <plantgl/scenegraph/container/pointmatrix.h>
#include <plantgl/scenegraph/core/pgl_messages.h>
#include <plantgl/math/util_math.h>

PGL_USING_NAMESPACE

/* ----------------------------------------------------------------------- */

Matrix4Transformation::Matrix4Transformation( ) :
  Transformation3D() {
}

Matrix4Transformation::~Matrix4Transformation( ) {
}


/* ----------------------------------------------------------------------- */

Matrix3Transformation::Matrix3Transformation( ) :
  Transformation2D() {
}

Matrix3Transformation::~Matrix3Transformation( ) {
}

/* ----------------------------------------------------------------------- */



MatrixTransformed::Builder::Builder() :
  Transformed::Builder(),
  Geometry(0) {
}


MatrixTransformed::Builder::~Builder() {
}


void MatrixTransformed::Builder::destroy() {
        MTDestroy() ;
}
void MatrixTransformed::Builder::MTDestroy() {
  if (Geometry) delete Geometry;
}


bool MatrixTransformed::Builder::isValid( ) const {
        return MTValid( ) ;
}
bool MatrixTransformed::Builder::MTValid( ) const {
  if (! Geometry) {
    pglErrorEx(PGLWARNINGMSG(UNINITIALIZED_FIELD_ss),"Matrix Transformed","Geometry");
    return false;
  };
  if (! (*Geometry) ) {
    pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Matrix Transformed","Geometry","Must be a valid Geometry Object.");
    return false;
  };
  return true;
}


/* ----------------------------------------------------------------------- */


MatrixTransformed::MatrixTransformed( const GeometryPtr& geometry ) :
  Transformed(),
  __geometry(geometry) {
}

MatrixTransformed::MatrixTransformed() :
  Transformed(),
  __geometry() {
}

MatrixTransformed::~MatrixTransformed( ) {
}

const GeometryPtr
MatrixTransformed::getGeometry( ) const {
  return __geometry;
}

GeometryPtr&
MatrixTransformed::getGeometry( ){
  return __geometry;
}

bool
MatrixTransformed::isACurve( ) const {
  return __geometry->isACurve();
}

bool
MatrixTransformed::isASurface( ) const {
  return __geometry->isASurface();
}

bool
MatrixTransformed::isAVolume( ) const {
  return __geometry->isAVolume();
}

bool
MatrixTransformed::isExplicit( ) const {
  return __geometry->isExplicit();
}

/* ----------------------------------------------------------------------- */
GeneralMatrix3Transformation::GeneralMatrix3Transformation( const Matrix3& mat ) :
  Matrix3Transformation(),
  __matrix(mat){
}

GeneralMatrix3Transformation::~GeneralMatrix3Transformation( ) {
}

Matrix3 GeneralMatrix3Transformation::getMatrix( ) const{
  return __matrix;
}

bool GeneralMatrix3Transformation::isValid() const {
    return __matrix.isValid();
}

Point2ArrayPtr
GeneralMatrix3Transformation::transform( const Point2ArrayPtr& points ) const{
  GEOM_ASSERT(points);
  Point2ArrayPtr _tPoints(new Point2Array(points->size()));
  Point2Array::iterator _ti = _tPoints->begin();
  for (Point2Array::const_iterator _i = points->begin();
       _i != points->end();
         _i++)
    *_ti++ = (__matrix * Vector3(*_i,1.0)).project();
  return _tPoints;
}

Point3ArrayPtr
GeneralMatrix3Transformation::transform( const Point3ArrayPtr& points ) const{
  GEOM_ASSERT(points);
  Point3ArrayPtr _tPoints(new Point3Array(points->size()));
  Point3Array::iterator _ti = _tPoints->begin();
  for (Point3Array::const_iterator _i = points->begin();
       _i != points->end();
         _i++)
    *_ti++ = __matrix * (*_i);
  return _tPoints;
}

Point2MatrixPtr
GeneralMatrix3Transformation::transform( const Point2MatrixPtr& points ) const{
  GEOM_ASSERT(points);
  Point2MatrixPtr _tPoints(new Point2Matrix(points->size()));
  Point2Matrix::iterator _ti = _tPoints->begin();
  for (Point2Matrix::const_iterator _i = points->begin();
       _i != points->end();
         _i++)
    *_ti++ = (__matrix * Vector3(*_i,1.0)).project();
  return _tPoints;
}

Point3MatrixPtr
GeneralMatrix3Transformation::transform( const Point3MatrixPtr& points ) const{
  GEOM_ASSERT(points);
  Point3MatrixPtr _tPoints(new Point3Matrix(points->size()));
  Point3Matrix::iterator _ti = _tPoints->begin();
  for (Point3Matrix::const_iterator _i = points->begin();
       _i != points->end();
         _i++)
    *_ti++ = __matrix * (*_i);
  return _tPoints;
}


/* ----------------------------------------------------------------------- */


Transform4::Builder::Builder() :
  Translation(0),
  Scale(0),
  Rotation(0)
{
  GEOM_TRACE("Constructor Transform4::Builder");
}

Transform4::Builder::~Builder(){
}

Transform4Ptr Transform4::Builder::build( ) const
{
  GEOM_TRACE("build Transform4::Builder");
  if( isValid() )
    {
    Transform4* T= new Transform4();
    if( Scale )
      T->scale(*Scale);
    if( Rotation )
      T->rotate((*Rotation)->getMatrix3());
    if( Translation )
      T->translate(*Translation);

    return Transform4Ptr(T);
    }

  return Transform4Ptr();
}

void Transform4::Builder::destroy( )
{
  GEOM_TRACE("destroy Transform4::Builder");
  if( Translation ) delete Translation; Translation= 0;
  if( Scale ) delete Scale; Scale= 0;
  if( Rotation ) delete Rotation; Rotation= 0;
}

bool Transform4::Builder::isValid( ) const
{
  GEOM_TRACE("validate Transform4::Builder");
  if( (!Translation) && (!Scale) && (!Rotation) )
    {
    pglErrorEx(PGLWARNINGMSG(UNINITIALIZED_FIELD_ss),"Transform4","Translation, Scale and Rotation");
    return false;
    }

  if( Translation && (!Translation->isValid()) )
    {
    pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Transform4","Translation","Must be a valid Translation.");
    return false;
    }

  if( Scale && (!Scale->isValid()) )
    {
    pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Transform4","Scale","Must be a valid Scale.");
    return false;
    }

  if( Rotation && (!(*Rotation)->isValid()) )
    {
    pglErrorEx(PGLWARNINGMSG(INVALID_FIELD_VALUE_sss),"Transform4","Rotation","Must be a valid Rotation.");
    return false;
    }

  return true;
}

/* ----------------------------------------------------------------------- */

Transform4::Transform4() :
  __matrix(Matrix4::IDENTITY) {
}

Transform4::Transform4( const Matrix4& mat ) :
  __matrix(mat){
}

Transform4::~Transform4( ) {
}

Matrix4 Transform4::getMatrix( ) const{
  return __matrix;
}

Matrix4& Transform4::getMatrix( ){
  return __matrix;
}

/* ----------------------------------------------------------------------- */

/////////////////////////////////////////////////////////////////////////////
bool Transform4::isValid() const
/////////////////////////////////////////////////////////////////////////////
{
  return __matrix.isValid();
}

/////////////////////////////////////////////////////////////////////////////
Point3ArrayPtr Transform4::transform( const Point3ArrayPtr& points ) const
/////////////////////////////////////////////////////////////////////////////
{
  GEOM_ASSERT(points);
  Point3ArrayPtr _tPoints(new Point3Array(points->size()));
  Point3Array::iterator _ti = _tPoints->begin();
  Point3Array::const_iterator _i = points->begin();
  for ( _i = points->begin(); _i != points->end(); _i++ )
    *_ti++ = __matrix * (*_i);
  return _tPoints;
}

/////////////////////////////////////////////////////////////////////////////
Point4ArrayPtr Transform4::transform( const Point4ArrayPtr& points ) const
/////////////////////////////////////////////////////////////////////////////
{
  GEOM_ASSERT(points);
  Point4ArrayPtr _tPoints(new Point4Array(points->size()));
  Point4Array::iterator _ti = _tPoints->begin();
  Point4Array::const_iterator _i = points->begin();
  for ( _i = points->begin(); _i != points->end(); _i++ )
    *_ti++ = __matrix * (*_i);
  return _tPoints;
}

/////////////////////////////////////////////////////////////////////////////
Point3MatrixPtr Transform4::transform( const Point3MatrixPtr& points ) const
/////////////////////////////////////////////////////////////////////////////
{
  GEOM_ASSERT(points);
  Point3MatrixPtr _tPoints(new Point3Matrix(points->size()));
  Point3Matrix::iterator _ti = _tPoints->begin();
  Point3Matrix::const_iterator _i = points->begin();
  for ( _i = points->begin(); _i != points->end(); _i++ )
    *_ti++ = __matrix * (*_i);
  return _tPoints;
}

/////////////////////////////////////////////////////////////////////////////
Point4MatrixPtr Transform4::transform( const Point4MatrixPtr& points ) const
/////////////////////////////////////////////////////////////////////////////
{
  GEOM_ASSERT(points);
  Point4MatrixPtr _tPoints(new Point4Matrix(points->size()));
  Point4Matrix::iterator _ti = _tPoints->begin();
  Point4Matrix::const_iterator _i = points->begin();
  for ( _i = points->begin(); _i != points->end(); _i++ )
    *_ti++ = __matrix * (*_i);
  return _tPoints;
}

/////////////////////////////////////////////////////////////////////////////
Transform4& Transform4::translate( const Vector3& t )
/////////////////////////////////////////////////////////////////////////////
{
  Matrix4 tm= Matrix4::translation(t);
  __matrix= tm * __matrix;
//cout<<"translation :"<<tm<<endl;
//cout<<"result :"<<__matrix<<endl<<endl;;
  return *this;
}

/////////////////////////////////////////////////////////////////////////////
Transform4& Transform4::scale( const Vector3& s )
/////////////////////////////////////////////////////////////////////////////
{
  Matrix4 sm(Matrix3::scaling(s));
  __matrix= sm * __matrix;
//cout<<"scaling :"<<sm<<endl;
//cout<<"result :"<<__matrix<<endl<<endl;;
  return *this;
}

/////////////////////////////////////////////////////////////////////////////
Transform4& Transform4::rotate( const Matrix3& m )
/////////////////////////////////////////////////////////////////////////////
{
  Matrix4 rm(m);
  __matrix= rm * __matrix;
//cout<<"rotated :"<<rm<<endl;
//cout<<"result :"<<__matrix<<endl<<endl;;
  return *this;
}

/////////////////////////////////////////////////////////////////////////////
void Transform4::getTransformation( Vector3& scale,
                                    Vector3& rotate,
                                    Vector3& translate )
/////////////////////////////////////////////////////////////////////////////
{
  __matrix.getTransformation(scale,rotate,translate);
}

/////////////////////////////////////////////////////////////////////////////
real_t Transform4::getVolume() const
/////////////////////////////////////////////////////////////////////////////
{
  return det( Matrix3(__matrix) );
}
