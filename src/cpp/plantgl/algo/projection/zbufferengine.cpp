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
             


/* ----------------------------------------------------------------------- */

#include "zbufferengine.h"
#include "projectionrenderer.h"
#include "projection_util.h"
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <chrono>
#include <array>
#include <atomic>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
/* ----------------------------------------------------------------------- */

PGL_USING_NAMESPACE

ZBufferEngine::ZBufferEngine(uint16_t imageWidth, 
                             uint16_t imageHeight,
                             eRenderingStyle style,
                             const Color3& backGroundColor,
                             uint32_t defaultid,
                             bool multithreaded,
                             eFaceCulling culling,
                             eIdPolicy idPolicy ):
    ImageProjectionEngine(imageWidth,imageHeight,idPolicy),
    __light(new Light()),
    __style(style),
    __defaultid(defaultid),
    __alphathreshold(0.99),
    __depthBuffer(new RealArray2(uint_t(imageWidth), uint_t(imageHeight), REAL_MAX)),
    __frameBuffer((style & eColorBased) ? new PglFrameBufferManager(imageWidth, imageHeight, 3, backGroundColor) : NULL),
    __idBuffer((style & eIdBased) ? new Uint32Array2(uint_t(imageWidth), uint_t(imageHeight), defaultid) : NULL),
    __orientationBuffer((style & eOrientationBased) ? new BoolArray2(uint_t(imageWidth), uint_t(imageHeight), defaultid) : NULL),
    __imageMutex(),
    __triangleshader((style != eDepthOnly) ? new TriangleShaderSelector(this) : NULL),
    __triangleshaderset(NULL),
    __multithreaded(multithreaded),
    __culling(culling)
{
    beginProcess();
}    

    
ZBufferEngine::~ZBufferEngine()
{
}

void ZBufferEngine::lock(uint_t x, uint_t y)
{
    if(__multithreaded) __imageMutex->lock(x,y);
}

void ZBufferEngine::unlock(uint_t x, uint_t y)
{
    if(__multithreaded) __imageMutex->unlock(x,y);    
}

bool ZBufferEngine::tryLock(uint_t x, uint_t y)
{
    if(__multithreaded) return __imageMutex->tryLock(x,y);
    return true;   
}

void ZBufferEngine::beginProcess()
{

    if (__multithreaded && is_null_ptr(__imageMutex)){
        // printf("begin rendering : create thread pool\n");
        __imageMutex = getImageMutex(__imageWidth, __imageHeight);
    }

}

void ZBufferEngine::endProcess()
{

    if(__multithreaded){
        ThreadManager::get().join();
    }
}

void ZBufferEngine::setHemisphericCamera(real_t nearValue, real_t farValue)
{  
    setSphericalCamera(180, nearValue, farValue);
}

void ZBufferEngine::setSphericalCamera(real_t viewAngle, real_t nearValue, real_t farValue)
{  
    __camera = ProjectionCamera::sphericalCamera(viewAngle, nearValue, farValue); 
    for (uint32_t x = 0; x < __imageWidth; ++x){
        for (uint32_t y = 0; y < __imageHeight; ++y){
            if (!__camera->isValidPixel(x,y,__imageWidth,__imageHeight)){
                if(__style & eColorBased) setFrameBufferAt(x,y,Color3::BLACK);
                __depthBuffer->setAt(x, y, nearValue);
            }
        }
    }
}
void ZBufferEngine::setEquirectangularCamera(real_t viewAngle, real_t nearValue, real_t farValue)
{  
    __camera = ProjectionCamera::equirectangularCamera(viewAngle, nearValue, farValue); 
}

void ZBufferEngine::setCylindricalCamera(real_t viewAngle, real_t bottom, real_t top, real_t nearValue, real_t farValue)
{  
    __camera = ProjectionCamera::cylindricalCamera(viewAngle, bottom, top, nearValue, farValue); 
}


ImagePtr ZBufferEngine::getImage() const
{
    PglFrameBufferManagerPtr fb = dynamic_pointer_cast<PglFrameBufferManager>(__frameBuffer);
    if (is_valid_ptr(fb)) { return fb->getImage(); }
    printf("No valid framebuffer\n");
    return ImagePtr();
}

ImagePtr ZBufferEngine::getIdBufferAsImage(Color4::eColor4Format conversionformat) const
{
    ImagePtr result(new Image(__imageWidth, __imageHeight, 4));

    for (uint_t x = 0 ; x < __imageWidth ; ++x)
        for (uint_t y = 0 ; y < __imageHeight ; ++y){
            result->setPixelAt(x,y,Color4::fromUint(__idBuffer->getAt(x,y), conversionformat));
        }

    return result;
}


bool ZBufferEngine::isTotallyTransparent(const real_t alpha) const
{
    return alpha >= __alphathreshold;
}


Color3 ZBufferEngine::getFrameBufferAt(uint32_t x, uint32_t y) const
{
    if(__frameBuffer) return __frameBuffer->getPixelAt(x,y);
    return Color3(0,0,0);
}

void ZBufferEngine::setFrameBufferAt(uint32_t x, uint32_t y, const Color3& rasterColor)
{
    if(__frameBuffer) __frameBuffer->setPixelAt(x,y,rasterColor);
}

void ZBufferEngine::setFrameBufferAt(uint32_t x, uint32_t y, const Color4& rasterColor)
{
    if (rasterColor.getAlpha() < GEOM_EPSILON )
        setFrameBufferAt(x,y, Color3(rasterColor));
    else if (!isTotallyTransparent(rasterColor.getAlpha())) {
        real_t alpha = rasterColor.getAlphaClamped();
        Color3 current = getFrameBufferAt(x,y);
        Color3 result = current * (1-alpha) + rasterColor * alpha;
        setFrameBufferAt(x,y, result);
    }
}

bool ZBufferEngine::isVisible(int32_t x, int32_t y, real_t z) const
{
    real_t cz = __depthBuffer->getAt(x, y);
    return (z < cz && (cz-z) > GEOM_EPSILON); 
}

bool ZBufferEngine::isVisible(const Vector3& pos) const
{
    Vector3 deltapos = pos - camera()->position();
    Vector3 direction = - camera()->direction();
    real_t hdistancetocamera = dot(deltapos, direction);
    real_t nearValue = camera()->get_near();

    if (hdistancetocamera < nearValue){
        // printf("Position (%f,%f,%f) is behind near plane : %f\n", pos.x(), pos.y(), pos.z(), (dot(pos-camera()->position(), -camera()->getWorldToCameraMatrix().getColumn(2)) - camera()->near));
        return false;
    }
    Vector3 raster = worldToRaster(pos);
    // printf("Position (%f,%f,%f) isVisible : (%i,%i), (%f,%f)  %f %f %s\n", pos.x(), pos.y(), pos.z(), int(raster.x()), int(raster.y()), raster.x(), raster.y(), raster.z(), __depthBuffer->getAt(raster.x(), raster.y()), (raster.z()< __depthBuffer->getAt(raster.x(), raster.y()) ? "True" : "False"));
    return isVisible(raster.x(), raster.y(), raster.z());
}


bool ZBufferEngine::renderRaster(uint32_t x, uint32_t y, real_t z, const Color4& rasterColor, const uint32_t id)
{
    if (isTotallyTransparent(rasterColor.getAlpha())) return false;

    lock(x,y);
    if (isVisible(x,y,z)) {
        __depthBuffer->setAt(x, y, z);
        if(__style & eColorBased){
            setFrameBufferAt(x, y, rasterColor);
        }
        if(__style & eIdBased){
            __idBuffer->setAt(x, y, id);
        }
        unlock(x,y);
        return true;
    }
    unlock(x,y);
    return false;
}

void ZBufferEngine::_tryRenderRaster(const struct Fragment& fragment, FragmentQueue& failqueue)
{
    if (!_tryRenderRaster(fragment.x, fragment.y, fragment.z, fragment.color, fragment.id, fragment.orientation))
        failqueue.push(fragment);
}

bool ZBufferEngine::_tryRenderRaster(uint32_t x, uint32_t y, real_t z, const Color4& rasterColor, const uint32_t id, const bool orientation)
{
    if (isTotallyTransparent(rasterColor.getAlpha())) return false;

    if(tryLock(x,y)){
        if (isVisible(x,y,z)) {
            __depthBuffer->setAt(x, y, z);
            if(__style & eColorBased){
                setFrameBufferAt(x, y, rasterColor);
            }
            if(__style & eIdBased){
                __idBuffer->setAt(x, y, id);
            }
            if(__style & eOrientationBased){
                __idBuffer->setAt(x, y, orientation);
            }
        }
        unlock(x,y);
        return true;
    }
    else return false;

}

void ZBufferEngine::setLight(const Vector3& lightPosition, const Color3& lightColor, bool directional)
{
    __light->set(lightPosition, lightColor, directional);
}

void ZBufferEngine::setLight(const Vector3& lightPosition, const Color3& lightAmbient, const Color3& lightDiffuse, const Color3& lightSpecular, bool directional)
{
    __light->set(lightPosition, lightAmbient, lightDiffuse, lightSpecular, directional);
}


void ZBufferEngine::iprocess(TriangleSetPtr triangles, AppearancePtr appearance, uint32_t id, ProjectionCameraPtr _camera, uint32_t threadid)
{
    // printf("process TriangleSetPtr\n");
    const Point3ArrayPtr points(triangles->getPointList());
    const Index3ArrayPtr indices(triangles->getIndexList());

    bool ccw = triangles->getCCW();

    triangles->checkNormalList();

    size_t nbfaces = triangles->getIndexListSize();
    bool hasColor = triangles->hasColorList();

    TriangleShaderPtr shader;
    if (getRenderingStyle() & eColorBased){
        if (threadid != 0) {
            assert(threadid <= ThreadManager::get().nb_threads());
            //printf("session %u\n", threadid);
            shader = __triangleshaderset[threadid-1];
        }
        else {
            shader = __triangleshader;
        }
    }

    // ProjectionCameraPtr camera = (__multithreaded?ProjectionCameraPtr(_camera->copy()):_camera);
    for(uint32_t itidx = 0; itidx < nbfaces; ++itidx){
        //printf("triangle %i on %i\n", itidx, nbfaces);

        const Vector3& v0 = triangles->getFacePointAt(itidx,0);
        const Vector3& v1 = triangles->getFacePointAt(itidx,1);
        const Vector3& v2 = triangles->getFacePointAt(itidx,2);

        if(is_valid_ptr(shader)){
            shader->init(appearance, triangles, itidx, _camera, __light);
        }
      
        // shader->initEnv(_camera, __light);
        renderShadedTriangle(v0, v1, v2, ccw, getNewPrimitiveId(id), shader, _camera);

    }
    // printf("end process TriangleSetPtr\n");

}


// Bound a spherical projection using a cap containing all triangle rays.
static void sphericalRasterBounds(const SphericalCamera& camera,
                                  const Vector3& a, const Vector3& b, const Vector3& c,
                                  uint16_t width, uint16_t height,
                                  real_t& xmin, real_t& ymin, real_t& xmax, real_t& ymax)
{
    xmin = ymin = 0;
    xmax = width;
    ymax = height;
    const real_t halfAngle = camera.getViewAngle() * GEOM_RAD / 2;
    if (halfAngle <= GEOM_EPSILON || halfAngle > GEOM_HALF_PI ||
        norm(a) <= GEOM_EPSILON || norm(b) <= GEOM_EPSILON || norm(c) <= GEOM_EPSILON)
        return;

    const Vector3 u = a.normed(), v = b.normed(), w = c.normed();
    const Vector3 sum = u + v + w;
    if (norm(sum) <= GEOM_EPSILON) return;
    const Vector3 center = sum.normed();
    if (center.z() >= 0) return;

    const real_t chord = max3(norm(u - center), norm(v - center), norm(w - center));
    const real_t angularRadius = 2 * std::asin(std::min(real_t(1), chord / 2)) + GEOM_EPSILON;
    if (angularRadius >= GEOM_HALF_PI) return;
    // A cap of angular radius below pi/2 is convex in ray directions.
    // On the forward hemisphere, the equidistant map expands angular
    // distances by at most pi/2 before scaling by the camera's half angle.
    const real_t radius = angularRadius * GEOM_HALF_PI / halfAngle;
    const Vector3 rasterCenter = camera.cameraToRaster(center, width, height);
    const real_t rx = radius * width / 2;
    const real_t ry = radius * height / 2;
    xmin = std::max(real_t(0), rasterCenter.x() - rx);
    xmax = std::min(real_t(width), rasterCenter.x() + rx);
    ymin = std::max(real_t(0), rasterCenter.y() - ry);
    ymax = std::min(real_t(height), rasterCenter.y() + ry);
}

void ZBufferEngine::renderShadedTriangle(const TOOLS(Vector3)& v0, const TOOLS(Vector3)& v1_, const TOOLS(Vector3)& v2_, bool ccw, const uint32_t id, const TriangleShaderPtr& shader,  const ProjectionCameraPtr& camera)
{
    Vector3 v1 = v1_;
    Vector3 v2 = v2_;
    if (ccw == false) {
        v1 = v2_;
        v2 = v1_;
    }

     // Projection in camera space
    Vector3 v0Cam = camera->worldToCamera(v0);
    Vector3 v1Cam = camera->worldToCamera(v1);
    Vector3 v2Cam = camera->worldToCamera(v2);

    const bool spherical = camera->type() == ProjectionCamera::eHemispheric;
    Vector3 v0Raster, v1Raster, v2Raster;
    if (!spherical) {
        v0Raster = camera->cameraToRaster(v0Cam, __imageWidth, __imageHeight);
        v1Raster = camera->cameraToRaster(v1Cam, __imageWidth, __imageHeight);
        v2Raster = camera->cameraToRaster(v2Cam, __imageWidth, __imageHeight);
    }

    /*printf("rasterize t point [%f,%f,%f] \n",v0Raster.x(),v0Raster.y(),v0Raster.z());
    printf("rasterize t point [%f,%f,%f] \n",v1Raster.x(),v1Raster.y(),v1Raster.z());
    printf("rasterize t point [%f,%f,%f] \n",v2Raster.x(),v2Raster.y(),v2Raster.z());*/

    v0Raster.x() = std::round(v0Raster.x());
    v0Raster.y() = std::round(v0Raster.y());
    v1Raster.x() = std::round(v1Raster.x());
    v1Raster.y() = std::round(v1Raster.y());
    v2Raster.x() = std::round(v2Raster.x());
    v2Raster.y() = std::round(v2Raster.y());

    real_t xmin, ymin, zmin;
    real_t xmax, ymax, zmax;

    xmin = min3(v0Raster.x(), v1Raster.x(), v2Raster.x());
    ymin = min3(v0Raster.y(), v1Raster.y(), v2Raster.y());

    xmax = max3(v0Raster.x(), v1Raster.x(), v2Raster.x());
    ymax = max3(v0Raster.y(), v1Raster.y(), v2Raster.y());

    zmin = min3(v0Raster.z(), v1Raster.z(), v2Raster.z());
    zmax = max3(v0Raster.z(), v1Raster.z(), v2Raster.z());

    if (spherical) {
        sphericalRasterBounds(static_cast<const SphericalCamera&>(*camera), v0Cam, v1Cam, v2Cam,
                              __imageWidth, __imageHeight, xmin, ymin, xmax, ymax);
    }
    else if(camera->methodType() == ProjectionCamera::eRayIntersection && (std::floor(xmax)-std::floor(xmin)>1) && (std::floor(ymax)-std::floor(ymin)>1)) {
        std::pair<Vector3, real_t> csphere = circumsphere(v0Raster,v1Raster,v2Raster);
        Vector3 vMeanRaster = csphere.first;
        real_t radius = csphere.second;

        xmin = vMeanRaster.x() - radius;
        ymin = vMeanRaster.y() - radius;

        xmax = vMeanRaster.x() + radius;
        ymax = vMeanRaster.y() + radius;

        if (zmin < 0 && zmax > 0) {
            if (xmax < __imageWidth/2){ xmin = 0;}
            else if  (xmin > __imageWidth/2){ xmax = __imageWidth;}
        }

    }

      // the triangle is out of screen
    if (xmin >= __imageWidth  || xmax < 0 || ymin >= __imageHeight || ymax < 0 || (!spherical && !camera->isInZRange(zmin, zmax))) {
        return;
    }

    // be careful xmin/xmax/ymin/ymax can be negative. Don't cast to uint32_t
    int32_t x0 = pglMax(int32_t(0), (int32_t)(std::floor(xmin)));
    int32_t x1 = pglMin(int32_t(__imageWidth) - 1, (int32_t)(std::ceil(xmax)));
    int32_t y0 = pglMax(int32_t(0), (int32_t)(std::floor(ymin)));
    int32_t y1 = pglMin(int32_t(__imageHeight) - 1, (int32_t)(std::ceil(ymax)));

    if (__multithreaded && (x1-x0+1)*(y1-y0+1) > 20) {
        std::tuple<Vector3,Vector3,Vector3> vRasters(v0Raster, v1Raster, v2Raster);
        std::tuple<Vector3,Vector3,Vector3> vCams(v0Cam,v1Cam,v2Cam);

        ThreadManager::get().new_task(boost::bind(&ZBufferEngine::rasterizeMT, this, Index4(x0,x1,y0,y1), 
                                                  // v0Raster, v1Raster, v2Raster, v0Cam,v1Cam,v2Cam,
                                                  vRasters, vCams, id, 
                                                  (getRenderingStyle() & eColorBased) ? TriangleShaderPtr(shader->copy()) : shader, ProjectionCameraPtr(camera->copy())));
    }
    else {
        rasterize(x0,x1,y0,y1,v0Raster,v1Raster,v2Raster,v0Cam,v1Cam,v2Cam,id,shader,camera);
    }
}



void ZBufferEngine::rasterizeMT(const Index4& rect,
                                const std::tuple<Vector3,Vector3,Vector3>& vRasters, 
                                const std::tuple<Vector3,Vector3,Vector3>& vCams,
                                const uint32_t id, 
                                const TriangleShaderPtr& shader, const ProjectionCameraPtr& camera)
{
    rasterize(rect[0],rect[1],rect[2],rect[3],
              std::get<0>(vRasters),std::get<1>(vRasters),std::get<2>(vRasters),
              std::get<0>(vCams),std::get<1>(vCams),std::get<2>(vCams),
             // v1Raster,v2Raster,v0Cam,v1Cam,v2Cam,
              id,shader,camera);  
}

void findorder(real_t * zs, int& firstpoint, int& secpoint, int& thirdpoint){
    firstpoint = 0; secpoint = 1; thirdpoint = 2;
    if (zs[firstpoint] < zs[secpoint]) std::swap(firstpoint, secpoint);
    if (zs[secpoint] < zs[thirdpoint]) std::swap(thirdpoint, secpoint);
    if (zs[firstpoint] < zs[secpoint]) std::swap(firstpoint, secpoint);
}

ZBufferEngine::eFaceIntersection ZBufferEngine::intersectionSignTest(real_t w0, real_t w1, real_t w2) const {
    if (w0 > -GEOM_EPSILON && w1 > -GEOM_EPSILON && w2 > -GEOM_EPSILON) return eFrontFaceIntersection;
    else if (w0 < GEOM_EPSILON && w1 < GEOM_EPSILON && w2 < GEOM_EPSILON) return eBackFaceIntersection;
    else return eNoIntersection;
}

ZBufferEngine::eFaceIntersection ZBufferEngine::culling(ZBufferEngine::eFaceIntersection intersection) const {
    if (__culling == eBothFaceCulling)
        return eNoIntersection;
    else if (__culling == eFrontFaceCulling && intersection == eFrontFaceIntersection)
        return eNoIntersection;
    else if (__culling == eBackFaceCulling && intersection == eBackFaceIntersection)
        return eNoIntersection;
    return intersection;
}

void ZBufferEngine::rasterize(int32_t x0, int32_t x1, int32_t y0, int32_t y1,
                              TOOLS(Vector3) v0Raster, TOOLS(Vector3) v1Raster, TOOLS(Vector3) v2Raster, 
                              TOOLS(Vector3) v0Cam, TOOLS(Vector3) v1Cam, TOOLS(Vector3) v2Cam, const uint32_t id, 
                              const TriangleShaderPtr& shader, const ProjectionCameraPtr& camera)
{

    if (__culling == ZBufferEngine::eBothFaceCulling) return;

    real_t z0 = v0Raster.z();
    real_t z1 = v1Raster.z();
    real_t z2 = v2Raster.z();

    Vector3 normal = cross(v1Cam-v0Cam, v2Cam-v0Cam);

    if (camera->methodType() == ProjectionCamera::eProjection){
        bool orientation = normal.z() > -GEOM_EPSILON;
        if ( __culling == eFrontFaceCulling && orientation) return;
        else if ( __culling == eBackFaceCulling && !orientation) return;
    }

    // Precompute area of triangle
    real_t area;
    if (camera->methodType() == ProjectionCamera::eRayIntersection){
        area = norm(normal);
    }
    else {
        v0Raster.z() = 1. / v0Raster.z();
        v1Raster.z() = 1. / v1Raster.z();
        v2Raster.z() = 1. / v2Raster.z();

        area = edgeFunction(v0Raster, v1Raster, v2Raster);
    }

    FragmentQueue fragqueue;

    const bool spherical = camera->type() == ProjectionCamera::eHemispheric;

    // Inner loop
    if (!spherical && x0 == x1 && y0 == y1) {
        if (camera->isValidPixel(x0,y0,__imageWidth, __imageHeight)){
            real_t z = (z0+z1+z2)/3;
            real_t w0 = 1/3.;
            _tryRenderRaster(Fragment(x0,y0,z, is_valid_ptr(shader) ? shader->process(x0, y0, z, (w0 * z / z0), (w0 * z / z1), (w0 * z / z2)) : Color4::BLACK, id, true), fragqueue );
        }
    }
    else {
        for (int32_t y = y0; y <= y1; ++y) {
            for (int32_t x = x0; x <= x1; ++x) {

                Vector2 pixelSample(x + 0.5, y + 0.5);
                if (camera->methodType() == ProjectionCamera::eProjection){

                    // find weight of pixel
                    real_t w0 = edgeFunction(v1Raster, v2Raster, pixelSample);
                    real_t w1 = edgeFunction(v2Raster, v0Raster, pixelSample);
                    real_t w2 = edgeFunction(v0Raster, v1Raster, pixelSample);

                    // printf("%i %i : %f %f %f\n", x,y,w0,w1,w2);

                    eFaceIntersection intersection = culling(intersectionSignTest(w0,w1,w2));

                    if (intersection != eNoIntersection){
                        w0 /= area;
                        w1 /= area;
                        w2 /= area;

                        real_t oneOverZ = v0Raster.z() * w0 + v1Raster.z() * w1 + v2Raster.z() * w2;
                        real_t z = 1. / oneOverZ;

                        // Depth-buffer test
                        if (camera->isInZRange(z)){
                            _tryRenderRaster(Fragment(x,y,z, is_valid_ptr(shader) ? shader->process(x, y, z, (w0 * z / z0), (w1 * z / z1), (w2 * z / z2)) : Color4::BLACK, id, intersection == eFrontFaceIntersection), fragqueue );
                        }
                    }
                }
                else {
                    if (camera->isValidPixel(x, y, __imageWidth, __imageHeight)) {
                        const Vector3 sample(pixelSample.x(), pixelSample.y(), 1);
                        if (spherical) {
                            const Vector3 ndc = camera->rasterToNDC(sample, __imageWidth, __imageHeight);
                            if (ndc.x()*ndc.x() + ndc.y()*ndc.y() > 1) continue;
                        }
                        Vector3 intersection;
                        Ray ray = spherical
                            ? Ray(Vector3(0,0,0), camera->rasterToCamera(sample, __imageWidth, __imageHeight).normed())
                            : camera->rasterToCameraRay(x, y, __imageWidth, __imageHeight);
                        int intersected = ray.intersect(v0Cam, v1Cam, v2Cam, intersection);

                        bool orientation = dot(normal, ray.getDirection()) < GEOM_EPSILON;
                        if ( __culling == eFrontFaceCulling && orientation) continue;
                        else if ( __culling == eBackFaceCulling && !orientation) continue;

                        if (intersected == 1){
                            real_t w0 = norm(cross(v1Cam-v2Cam, v1Cam-intersection));
                            real_t w1 = norm(cross(v2Cam-v0Cam, v2Cam-intersection));
                            real_t w2 = norm(cross(v0Cam-v1Cam, v0Cam-intersection));
                            real_t z = norm(intersection);
                            if (spherical && !camera->isInZRange(z)) continue;
                            _tryRenderRaster(Fragment(x,y,z, is_valid_ptr(shader) ? shader->process(x, y, z, (w0 / area), (w1 / area), (w2 / area)) : Color4::BLACK, id, orientation), fragqueue );
                        }
                    }
                }
            }
        }
    }
    while(!fragqueue.empty()) {
        Fragment f = fragqueue.front();
        fragqueue.pop();
        _tryRenderRaster(f, fragqueue );
    }
}



void ZBufferEngine::renderTriangle(const Vector3& v0, const Vector3& v1, const Vector3& v2, 
                                     const Color4& c0,  const Color4& c1,  const Color4& c2, 
                                     bool ccw, const uint32_t id, ProjectionCameraPtr camera)
{
    if(is_null_ptr(camera)) camera = __camera;
    ColorBasedShader * shader = new ColorBasedShader(this);
    shader->setColors(c0, c1, c2);
    TriangleShaderPtr shaderptr(shader);
    renderShadedTriangle(v0, v1, v2, ccw, id, shaderptr, camera);
}


void ZBufferEngine::renderPoint(const TOOLS(Vector3)& v, const Color4& c, const uint32_t width, const uint32_t id, ProjectionCameraPtr camera)
{
    
    if(is_null_ptr(camera)) camera = __camera;

    Vector3 vCam = camera->worldToCamera(v);

    // Convert the vertices of the triangle to raster space
    Vector3 vRaster = camera->cameraToRaster(vCam,__imageWidth, __imageHeight);

    // printf("rasterize  point [%f,%f,%f] \n",vRaster.x(),vRaster.y(),vRaster.z());
    
    // v0Raster.z() = 1. / v0Raster.z();

    // the point is out of screen
    if (!camera->isValidPixel(vRaster.x(),vRaster.y(),__imageWidth, __imageHeight)||  
        !camera->isInZRange(vRaster.z())) {
        return;
    }

    int32_t halfwidth = width/2.;
    int32_t x0 = pglMax<int32_t>(int32_t(0), (int32_t)(vRaster.x()) - halfwidth );
    int32_t x1 = pglMin<int32_t>(int32_t(__imageWidth) - 1, (int32_t)(vRaster.x()) + halfwidth );
    int32_t y0 = pglMax<int32_t>(int32_t(0), (int32_t)(vRaster.y()) - halfwidth );
    int32_t y1 = pglMin<int32_t>(int32_t(__imageHeight) - 1, (int32_t)(vRaster.y()) + halfwidth );

    for (int32_t y = y0; y <= y1; ++y) {
        for (int32_t x = x0; x <= x1; ++x) {
            if (camera->isValidPixel(x,y,__imageWidth, __imageHeight)){
                renderRaster(x, y, vRaster.z(), c, id);
            }
        }
    }
}

void ZBufferEngine::renderSegment(const TOOLS(Vector3)& v0, const TOOLS(Vector3)& v1, const Color4& c0, const Color4& c1, const uint32_t width, const uint32_t id, ProjectionCameraPtr camera)
{
    if(is_null_ptr(camera)) camera = __camera;

    renderPoint(v0, c0, width, id, camera);
    renderPoint(v1, c1, width, id, camera);

    Vector3 v0Cam = camera->worldToCamera(v0);
    Vector3 v1Cam = camera->worldToCamera(v1);

    // Convert the vertices of the triangle to raster space
    Vector3 v0Raster = camera->cameraToRaster(v0Cam,__imageWidth, __imageHeight);
    Vector3 v1Raster = camera->cameraToRaster(v1Cam,__imageWidth, __imageHeight);

    if (fabs(v1Raster.y()-v0Raster.y()) < fabs(v1Raster.x()-v0Raster.x())){
        if(v0Raster.x() > v1Raster.x()){
            _renderSegment(0, v1Raster, v0Raster, c1, c0, width, id);
        }
        else{
            _renderSegment(0, v0Raster, v1Raster, c0, c1, width, id);
        }
    }
    else {
        if(v0Raster.y() > v1Raster.y()){
            _renderSegment(1, v1Raster, v0Raster, c1, c0, width, id);
        }
        else {
            _renderSegment(1, v0Raster, v1Raster, c0, c1, width, id);
        }
    }
}

void ZBufferEngine::_renderSegment(uchar dim, const TOOLS(Vector3)& v0Raster, const TOOLS(Vector3)& v1Raster, const Color4& c0, const Color4& c1, const uint32_t width, const uint32_t id)
{
    Vector3 dRaster = v1Raster - v0Raster;
    real_t totW = norm(dRaster);
    real_t Dz = dRaster.z()/dRaster[dim];
    uchar otherdim = (dim + 1) %2;
    uint32_t dims[2] = { __imageWidth , __imageHeight };

    real_t decal = 1;
    if (dRaster[otherdim] < 0){
        decal = -1;
        dRaster[otherdim] = -dRaster[otherdim];
    }

    real_t D = 2 * dRaster[otherdim] - dRaster[dim];
    Vector3 cRaster = v0Raster;

    int32_t halfwidth = width/2.;

    while (cRaster[dim] < v1Raster[dim]+1) {
        if ((0 <= cRaster[dim] && cRaster[dim] < dims[dim])&&
            (0 <= cRaster[otherdim] && cRaster[otherdim] < dims[otherdim])){
            real_t w = norm(cRaster - v0Raster) / totW;
            Color4 c = c0 * (1-w) + c1 * w;
            // printf("Render raster %f %f %f\n", cRaster.x(), cRaster.y(), cRaster.z());
            int32_t w0 = pglMax<int32_t>(int32_t(0), (int32_t)(cRaster[otherdim]) - halfwidth );
            int32_t w1 = pglMin<int32_t>(int32_t(dims[otherdim]) - 1, (int32_t)(cRaster[otherdim] + halfwidth ));
            Vector3 ciRaster = cRaster;
            for (int di = w0 ; di <= w1; ++di) {
                ciRaster[otherdim] = di;
                if (__camera->isValidPixel(ciRaster.x(),ciRaster.y(),__imageWidth, __imageHeight)){
                    renderRaster(ciRaster.x(), ciRaster.y(), ciRaster.z(), c, id);
                }
            }
        }
        if ( D > 0){
            cRaster[otherdim] += decal;
            D -= 2*dRaster[dim];
        }
        D += 2* dRaster[otherdim];
        cRaster.z() += Dz;
        cRaster[dim] += 1;
    }
}

void ZBufferEngine::iprocess(PolylinePtr polyline, MaterialPtr material, uint32_t id, ProjectionCameraPtr camera, uint32_t threadid)
{
    if(is_null_ptr(camera)) camera = __camera;
    Point3ArrayPtr points = polyline->getPointList();
    if (is_valid_ptr(points) && points->size() > 1){
        Color4ArrayPtr colorlist = polyline->getColorList();
        if (is_valid_ptr(colorlist)){
            for(uint32_t i = 0; i < points->size()-1; ++i){
                renderSegment(points->getAt(i), points->getAt(i+1), colorlist->getAt(i), colorlist->getAt(i+1), polyline->getWidth(), id, camera );
            }
        }
        else {
            Color4 color = Color4(material->getDiffuseColor(),material->getTransparency());
            for(uint32_t i = 0; i < points->size()-1; ++i){
                renderSegment(points->getAt(i), points->getAt(i+1), color, color, polyline->getWidth(), id, camera );
            }

        }
    }
}

void ZBufferEngine::iprocess(PointSetPtr pointset, MaterialPtr material, uint32_t id, ProjectionCameraPtr camera, uint32_t threadid)
{
    if(is_null_ptr(camera)) camera = __camera;

    const Point3ArrayPtr points(pointset->getPointList());
    Color4 defaultcolor = Color4(material->getDiffuseColor(),material->getTransparency());
    Color4Array::const_iterator itCol;
    bool colorPerPoint = pointset->hasColorList();
    if (colorPerPoint) {
        itCol = pointset->getColorList()->begin();
    }
    uint32_t pointsize = pointset->getWidth();
    for (Point3Array::const_iterator it = points->begin(); it!= points->end(); ++it)
    {
        renderPoint(*it, (colorPerPoint?*itCol:defaultcolor), pointsize, id, camera);
    }
}

ImagePtr ZBufferEngine::getTexture(const ImageTexturePtr imgdef)
{
    Cache<ImagePtr>::const_Iterator it = __cachetexture.find(imgdef->getObjectId());
    if (it != __cachetexture.end()){
        return it->second;
    }
    else {
        ImagePtr img(new Image(imgdef->getFilename()));
        __cachetexture.insert(imgdef->getObjectId(),img);
        return img;
    }
}


void ZBufferEngine::process(ScenePtr scene)
{
    beginProcess();
    size_t msize = scene->size();
    if(__multithreaded && msize > 100){
        size_t nbthreads = ThreadManager::get().nb_threads();
        size_t nbShapePerThread = (msize / nbthreads);
        if (nbShapePerThread * nbthreads < msize) { nbShapePerThread += 1; }

        if(__triangleshaderset != NULL) delete [] __triangleshaderset;
        __triangleshaderset = new TriangleShaderPtr[nbthreads];
        for (size_t j = 0 ; j < nbthreads ; ++j){
            __triangleshaderset[j] = TriangleShaderPtr(is_valid_ptr(__triangleshader) ? __triangleshader->copy(true) : NULL);
        }


        uint32_t threadid = 1;
        for (size_t i = 0 ; i < msize ; i+=nbShapePerThread, ++threadid) {
            
            Scene::const_iterator itbegin = scene->begin() + i ;
            Scene::const_iterator itend = scene->begin() + pglMin(msize, i + nbShapePerThread);

            ThreadManager::get().new_task(boost::bind(&ZBufferEngine::processScene, this, itbegin, itend, ProjectionCameraPtr(__camera->copy()),threadid));

            
        }
    } 
    else {
        processScene(scene->begin(), scene->end(), __camera);
    }
    endProcess();
}

void ZBufferEngine::processScene(Scene::const_iterator scene_begin, Scene::const_iterator scene_end, ProjectionCameraPtr camera, uint32_t threadid)
{
    Discretizer d;
    Tesselator t;
    ProjectionRenderer r(*this, camera, t, d, threadid);
    for (Scene::const_iterator it = scene_begin; it != scene_end; ++it)
        (*it)->apply(r);
}


void ZBufferEngine::duplicateBuffer(const TOOLS(Vector3)& from, const TOOLS(Vector3)& to, bool useDefaultColor, const Color3& defaultcolor)
{
    Vector3 vRasterFrom = __camera->worldToRaster(from,__imageWidth, __imageHeight);
    Vector3 vRasterTo = __camera->worldToRaster(to,__imageWidth, __imageHeight);
    Vector3 vDiff = vRasterTo - vRasterFrom;

    duplicateBuffer(vDiff.x(), vDiff.y(), vDiff.z(), useDefaultColor, defaultcolor);
}

void ZBufferEngine::duplicateBuffer(int32_t xDiff, int32_t yDiff, real_t zDiff, bool useDefaultColor, const Color3& defaultcolor){
    if(xDiff == 0  && yDiff == 0 && fabs(zDiff) < GEOM_EPSILON) return;

    uint32_t width = __imageWidth;
    uint32_t height = __imageHeight;

    uint32_t xStart = 0;
    uint32_t xEnd = width;

    if (xDiff > 0) { 
        if (xDiff > width) { return; }
        xStart += xDiff;
    }
    else {
        if (-xDiff > width) { return; }
        xEnd += xDiff;

    }

    uint32_t yStart = 0;
    uint32_t yEnd = height;
    if (yDiff > 0) { 
        if (yDiff > height) { return; }
        yStart += yDiff;
    }
    else {
        if (-yDiff > height) { return; }
        yEnd += yDiff;
    }

    RealArray2Ptr depthBuffer(new RealArray2(*__depthBuffer));
    FrameBufferManagerPtr framebuffer = (is_valid_ptr(__frameBuffer) ? __frameBuffer->deepcopy() : NULL);

    for (uint32_t i = xStart ; i < xEnd ; ++i) {
        for (uint32_t j = yStart ; j < yEnd ; ++j) {
            real_t potentialz = depthBuffer->getAt(i-xDiff,j-yDiff)+zDiff;
            if (__camera->isInZRange(potentialz))
                renderRaster(i, j, potentialz, useDefaultColor || is_null_ptr(framebuffer) ? defaultcolor : framebuffer->getPixelAt(i-xDiff,j-yDiff));
        }
    }

}

void ZBufferEngine::periodizeBuffer(const Vector3& from, const Vector3& to, bool useDefaultColor, const Color3& defaultcolor)
{
    Vector3 vRasterFrom = __camera->worldToRaster(from,__imageWidth, __imageHeight);
    Vector3 vRasterTo = __camera->worldToRaster(to,__imageWidth, __imageHeight);
    Vector3 vDiff = vRasterTo - vRasterFrom;

    periodizeBuffer(vDiff.x(), vDiff.y(), vDiff.z(), useDefaultColor, defaultcolor);
}

void ZBufferEngine::periodizeBuffer(int32_t xDiff, int32_t yDiff, real_t zDiff, bool useDefaultColor, const Color3& defaultcolor)
{
    // Check image bounds before negating the signed offsets.
    const int32_t width = __imageWidth, height = __imageHeight;
    if (xDiff <= -width || xDiff >= width || yDiff <= -height || yDiff >= height)
        return;
    _bufferPeriodizationStep(xDiff, yDiff, zDiff, useDefaultColor, defaultcolor);
    _bufferPeriodizationStep(-xDiff, -yDiff, -zDiff, useDefaultColor, defaultcolor);

}

void ZBufferEngine::_bufferPeriodizationStep(int32_t xDiff, int32_t yDiff, real_t zDiff, bool useDefaultColor, const Color3& defaultcolor)
{
    if(xDiff == 0  && yDiff == 0 && fabs(zDiff) < GEOM_EPSILON) return;

    const int32_t width = __imageWidth, height = __imageHeight;
    // Visit source pixels before their copies so a period can propagate to
    // the edge of the image in either direction.
    const int32_t xStart = xDiff < 0 ? width - 1 + xDiff : xDiff;
    const int32_t xEnd = xDiff < 0 ? -1 : width;
    const int32_t xStep = xDiff < 0 ? -1 : 1;
    const int32_t yStart = yDiff < 0 ? height - 1 + yDiff : yDiff;
    const int32_t yEnd = yDiff < 0 ? -1 : height;
    const int32_t yStep = yDiff < 0 ? -1 : 1;

    for (int32_t x = xStart; x != xEnd; x += xStep) {
        for (int32_t y = yStart; y != yEnd; y += yStep) {
            const int32_t sourceX = x - xDiff, sourceY = y - yDiff;
            const real_t sourceDepth = __depthBuffer->getAt(sourceX, sourceY);
            const real_t depth = sourceDepth + zDiff;
            if (sourceDepth == REAL_MAX || !__camera->isInZRange(depth) ||
                !__camera->isValidPixel(sourceX, sourceY, __imageWidth, __imageHeight) ||
                !__camera->isValidPixel(x, y, __imageWidth, __imageHeight))
                continue;
            renderRaster(x, y, depth, useDefaultColor || is_null_ptr(__frameBuffer)
                ? defaultcolor : getFrameBufferAt(sourceX, sourceY));
        }
    }
}


boost::asio::thread_pool * 
ThreadManager::getPool()
{
    if(__pool == NULL) __pool = new boost::asio::thread_pool(__nb_threads);
    return __pool;
}

ThreadManager::ThreadManager(): 
    __pool(NULL),
    __condition_lock(__condition_mutex),
    __nb_threads(boost::thread::hardware_concurrency()+1),
    __nb_tasks(0)
{}

ThreadManager::~ThreadManager()
{
    if(__pool != NULL) delete __pool;
}

void ThreadManager::new_task(std::function<void()> task){
    ++__nb_tasks;
    boost::asio::post(*getPool(), boost::bind(&ThreadManager::process_task, this, task));    
    //boost::asio::post(*__pool, task);    
}

void ThreadManager::process_task(std::function<void()> task)
{
    task();
    __threadend_mutex.lock();
    --__nb_tasks;
    __condition.notify_one();
    __threadend_mutex.unlock();
}

void ThreadManager::join()
{
    while(!hasCompletedTasks()){
        __condition.wait_for(__condition_lock, std::chrono::milliseconds(10), std::bind(&ThreadManager::hasCompletedTasks, this));
    }
    // __condition.wait(__condition_lock, std::bind(&ThreadManager::hasCompletedTasks, this));
}


bool ThreadManager::hasCompletedTasks() const { 
    // printf("rendering nb task : %u\n", uint32_t(__nb_tasks));
    return __nb_tasks == 0; 
}

// Singleton access
ThreadManager& ThreadManager::get() { return ThreadManager::THREADMANAGER; }

ThreadManager ThreadManager::THREADMANAGER;


ImageMutexPtr ZBufferEngine::IMAGEMUTEX;


ImageMutexPtr ZBufferEngine::getImageMutex(uint16_t imageWidth, uint16_t imageHeight)
{
    if (is_null_ptr(IMAGEMUTEX) ||  imageWidth > IMAGEMUTEX->width() || imageHeight > IMAGEMUTEX->height()){
        IMAGEMUTEX = ImageMutexPtr(new ImageMutex(pglMax<uint_t>(imageWidth, is_valid_ptr(IMAGEMUTEX)?IMAGEMUTEX->width():0), 
                                                  pglMax<uint_t>(imageHeight,is_valid_ptr(IMAGEMUTEX)?IMAGEMUTEX->height():0)));
    }
   return IMAGEMUTEX;
}


real_t pixweigth(const Vector2& a, const Vector2& b, real_t raywidth) {
    if (raywidth < 0.5) {
        if (norm(a-b) < GEOM_EPSILON) {
            return 1;
        }
        else return 0;
    }
    Vector2 c = abs(b-a);
    Vector2 d(*c.getMax(),*c.getMin());
    Vector2 dir(d);
    real_t n = dir.normalize();
    if (n <= GEOM_EPSILON) { return 1; }
    real_t n2 = norm(Vector2(0.5,dir.y()*0.5/dir.x()));
    if ((n+n2)<=raywidth) { return 1; }
    else if ((n-n2)>=raywidth) { return 0; }
    else {
        return (raywidth-(n-n2))/(2*n2);
    }
}

  std::tuple<PGL(Point3ArrayPtr),PGL(Color3ArrayPtr),PGL(Uint32Array1Ptr)> ZBufferEngine::grabZBufferPoints(real_t jitter, real_t raywidth) const
  {
   uint32_t width = __imageWidth;
   uint32_t height = __imageHeight;
   raywidth = fabs(raywidth);
   uint32_t raynbpix = ceil(raywidth);
   
   PGL(Point3ArrayPtr) points(new Point3Array());
   PGL(Color3ArrayPtr) colors;
   if (getRenderingStyle() & eColorBased)
        colors = new Color3Array();
   PGL(Uint32Array1Ptr) ids;
   if (getRenderingStyle() & eIdBased)
        ids = new Uint32Array1();
   
   Vector3 origin = __camera->position();
   for (int32_t i = 0 ; i < width ; i++) {
        for (int32_t j = 0 ; j < height ; j++) {
            real_t cz = 0;
            real_t count = 0;
            for (int ii = pglMax<int32_t>(0,i-raynbpix); ii <= pglMin<int32_t>(width-1,i+raynbpix); ++ii){
                for (int jj = pglMax<int32_t>(0,j-raynbpix); jj <= pglMin<int32_t>(height-1,j+raynbpix); ++jj){
                    real_t lcz = __depthBuffer->getAt(ii,jj);
                    if (__camera->isInZRange(lcz)) {
                        real_t w = pixweigth(Vector2(i,j),Vector2(ii,jj),raywidth);
                        if (w > GEOM_EPSILON) {
                            count += w;
                            cz += (lcz*w);
                        }
                    }
                }
            }
            if (count > GEOM_EPSILON) {
                cz /= count;
                Vector3 point = __camera->rasterToWorld(Vector3(i,j,cz), width, height);
                if (jitter > 0) {
                    Vector3 nml = point-origin;
                    real_t n = norm(nml);
                    if(n > GEOM_EPSILON){
                        nml /= n;
                        point += nml*jitter*(-1+2.*(rand()/double(RAND_MAX)));
                    }
                }
                if (point.isValid()) {
                    points->push_back(point);
                    if (getRenderingStyle() & eColorBased)
                        colors->push_back(getFrameBufferAt(i,j));
                    if (getRenderingStyle() & eIdBased){
                        uint32_t cid = __idBuffer->getAt(i,j);
                        if (cid == __defaultid){
                            typedef pgl_hash_map<int32_t,real_t> IdMap;
                            IdMap idcounts; 
                            for (int ii = pglMax<int32_t>(0,i-raynbpix); ii <= pglMin<int32_t>(width-1,i+raynbpix); ++ii){
                                for (int jj = pglMax<int32_t>(0,j-raynbpix); jj <= pglMin<int32_t>(height-1,j+raynbpix); ++jj){
                                    real_t lcz = __depthBuffer->getAt(ii,jj);
                                    if (__camera->isInZRange(lcz)) {
                                        real_t w = pixweigth(Vector2(i,j),Vector2(ii,jj),raywidth);
                                        if (w > GEOM_EPSILON) {
                                            uint32_t lid = __idBuffer->getAt(ii,jj);
                                            idcounts[lid] = (idcounts.find(lid) == idcounts.end()? 0 : idcounts[lid]) + w;
                                        }
                                    }
                                }
                            }
                            cid = idcounts.begin()->first;
                            real_t cw = idcounts.begin()->second;
                            for(IdMap::const_iterator itid = idcounts.begin() ; itid != idcounts.end() ; ++itid){
                                if (cw < itid->second){
                                    cw = itid->second;
                                    cid = itid->first;
                                }
                            }
                        }
                        ids->push_back(cid);
                    }
                }   
                else {
                    printf("invalid points\n");
                }
            }
        }
    }
    return  std::tuple<PGL(Point3ArrayPtr),PGL(Color3ArrayPtr),PGL(Uint32Array1Ptr)> (points, colors, ids);
  }

ScenePtr ZBufferEngine::grabSortedZBufferPoints(real_t jitter, real_t raywidth ) const
{
    ScenePtr result(new Scene());
    std::tuple<PGL(Point3ArrayPtr),PGL(Color3ArrayPtr),PGL(Uint32Array1Ptr)> pointinfos = grabZBufferPoints(jitter, raywidth);
    bool hasId = is_valid_ptr(std::get<2>(pointinfos));
    bool hasColor = is_valid_ptr(std::get<1>(pointinfos));
    if (!hasId) {
        Point3ArrayPtr points = std::get<0>(pointinfos);
        if(hasColor){
            Color4ArrayPtr colors = Color4ArrayPtr(new Color4Array(*std::get<1>(pointinfos)));
            result->add(ShapePtr(new Shape(GeometryPtr(new PointSet(points,colors)))));
        }
        else {
            result->add(ShapePtr(new Shape(GeometryPtr(new PointSet(points)))));
        }
    }
    else {
        typedef pgl_hash_map<uint32_t,PointSetPtr> PointSetMap;
        Point3Array::const_iterator piter = std::get<0>(pointinfos)->begin();
        Point3Array::const_iterator piterend = std::get<0>(pointinfos)->end();
        bool hasColor = is_valid_ptr(std::get<1>(pointinfos));
        Color3Array::const_iterator citer;
        if (hasColor) citer = std::get<1>(pointinfos)->begin();
        Uint32Array1::const_iterator iiter = std::get<2>(pointinfos)->begin();
        PointSetMap idmap;
        for(;piter != piterend; ++piter, ++citer, ++iiter){
            PointSetMap::const_iterator imiter = idmap.find(*iiter);
            PointSetPtr pset;
            if (imiter == idmap.end()){
                pset = PointSetPtr(new PointSet(new Point3Array(), (hasColor ? Color4ArrayPtr(new Color4Array()) : Color4ArrayPtr())));
                idmap[*iiter] = pset;
            }
            else { pset = imiter->second; }
            pset->getPointList()->push_back(*piter);
            if(hasColor) pset->getColorList()->push_back(*citer);
        }

        for(PointSetMap::const_iterator imiter = idmap.begin(); imiter != idmap.end(); ++imiter){
            result->add(ShapePtr(new Shape(GeometryPtr(imiter->second), AppearancePtr(0), imiter->first)));
        }
    }
    return result;
}


pgl_hash_map<uint32_t,real_t> ZBufferEngine::idhistogram(bool solidangle) const
{
    pgl_hash_map<uint32_t,real_t> histo;

    if ((__style & eIdBased) && is_valid_ptr(__idBuffer)){
       for (int32_t i = 0 ; i < __imageWidth ; i++) {
            for (int32_t j = 0 ; j < __imageHeight ; j++) {
            // for (Uint32Array2::const_iterator it = __idBuffer->begin(); it != __idBuffer->end(); ++it){
                uint32_t pid = __idBuffer->getAt(i,j);
                // printf("consider id %i\n",pid);
                if (pid != __defaultid){
                    pgl_hash_map<uint32_t,real_t>::iterator itId = histo.find(pid);
                    real_t value = (solidangle ? __camera->solidAngle(i,j,__imageWidth,__imageHeight) : 1);
                    if (itId == histo.end()){
                        histo[pid] = value;
                    }
                    else {
                        histo[pid] = itId->second+value;
                    }
                }
            }
        }
    }

    return histo;
}

RealArray2Ptr PGL(formFactors)(const Point3ArrayPtr& points, 
                               const Index3ArrayPtr& triangles, 
                               const Point3ArrayPtr& normals,
                               bool ccw,
                               uint16_t discretization,
                               bool solidangle)
{
    RealArray2Ptr result(new RealArray2(triangles->size(),triangles->size(),0));
    uint32_t id = 0;
    Color4 color = Color4::BLACK;
    for(Index3Array::const_iterator it = triangles->begin(); it != triangles->end(); ++it, ++id){
        const Vector3& v0 = points->getAt((*it)[0]); 
        const Vector3& v1 = points->getAt((*it)[1]); 
        const Vector3& v2 = points->getAt((*it)[2]); 
        Vector3 center = (v0+v1+v2)/3;
        Vector3 normal = (is_valid_ptr(normals) ? normals->getAt(id) : cross(v1-v0,v2-v0));
        if(normal.normalize() > GEOM_EPSILON){
            Vector3 up = v0-center;
            ZBufferEngine ze(discretization,discretization,ZBufferEngine::eIdBased, Color3::BLACK, Shape::NOID, true);
            ze.setHemisphericCamera();
            ze.lookAt(center, center+normal, up);
            uint32_t id2 = 0;
            ze.beginProcess();
            for(Index3Array::const_iterator it2 = triangles->begin(); it2 != triangles->end(); ++it2, ++id2){
                if (it2 != it){
                    ze.renderTriangle(points->getAt((*it2)[0]), points->getAt((*it2)[1]), points->getAt((*it2)[2]), 
                                      color, color, color, ccw, id2);
                }
            }
            ze.endProcess();
            pgl_hash_map<uint32_t,real_t> histo = ze.idhistogram(solidangle);
            for (pgl_hash_map<uint32_t,real_t>::const_iterator histoit = histo.begin(); histoit != histo.end(); ++histoit){
                result->setAt(id, histoit->first, histoit->second);
            }
        }
    }
    return result;
}


namespace {

void clipHemicubePolygon(std::vector<Vector3>& polygon, std::vector<Vector3>& clipped,
                         const Vector3& planeNormal, real_t offset)
{
    clipped.clear();
    Vector3 previous = polygon.back();
    real_t previousDistance = dot(planeNormal, previous) + offset;
    for (const auto& current : polygon) {
        const real_t distance = dot(planeNormal, current) + offset;
        if ((previousDistance < 0 && distance > 0) || (previousDistance > 0 && distance < 0)) {
            const real_t t = previousDistance / (previousDistance - distance);
            clipped.push_back(previous + (current - previous) * t);
        }
        if (distance >= 0) clipped.push_back(current);
        previous = current;
        previousDistance = distance;
    }
    polygon.swap(clipped);
}

void configureHemicubeFace(ZBufferEngine& buffer, bool topFace, real_t nearPlane, real_t farPlane)
{
    buffer.setFrustumCamera(-nearPlane, nearPlane, topFace ? -nearPlane : 0,
                            nearPlane, nearPlane, farPlane);
    buffer.lookAt(Vector3(0,0,0), Vector3(0,0,-1), Vector3(0,1,0));
}

void renderHemicubeFace(ZBufferEngine& buffer, const Point3Array& points,
                       const Index3Array& triangles, const Vector3& center,
                       const Vector3& direction, const Vector3& up,
                       bool topFace, uint32_t emitterId, real_t nearPlane, real_t farPlane,
                       Point3Array& cameraPoints, std::vector<Vector3>& polygon,
                       std::vector<Vector3>& clipped)
{
    const Vector3 right = cross(direction, up);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Vector3 relative = points.getAt(i) - center;
        cameraPoints.setAt(i, Vector3(dot(right, relative), dot(up, relative),
                                      -dot(direction, relative)));
    }

    const std::array<Vector3, 6> planes = {
        Vector3(0,0,-1), Vector3(0,0,1), Vector3(1,0,-1), Vector3(-1,0,-1),
        Vector3(0,1,topFace ? -1 : 0), Vector3(0,-1,-1)
    };
    const std::array<real_t, 6> offsets = {-nearPlane, farPlane, 0, 0, 0, 0};
    for (uint32_t id = 0; id < triangles.size(); ++id) {
        if (id == emitterId) continue;
        const auto& triangle = triangles.getAt(id);
        polygon = {cameraPoints.getAt(triangle[0]), cameraPoints.getAt(triangle[1]),
                   cameraPoints.getAt(triangle[2])};
        for (std::size_t i = 0; i < planes.size() && polygon.size() >= 3; ++i)
            clipHemicubePolygon(polygon, clipped, planes[i], offsets[i]);
        for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
            if (normSquared(cross(polygon[i] - polygon[0], polygon[i+1] - polygon[0])) > 0)
                buffer.renderShadedTriangle(polygon[0], polygon[i], polygon[i + 1], true, id,
                                            TriangleShaderPtr(), buffer.camera());
        }
    }
}

std::vector<real_t> hemicubePixelWeights(uint16_t resolution, bool topFace)
{
    const uint16_t height = topFace ? resolution : resolution / 2;
    std::vector<real_t> weights(std::size_t(resolution) * height);
    const real_t step = real_t(2) / resolution;
    for (uint16_t y = 0; y < height; ++y) {
        const real_t v = 1 - (y + real_t(0.5)) * step;
        for (uint16_t x = 0; x < resolution; ++x) {
            const real_t u = -1 + (x + real_t(0.5)) * step;
            const real_t squaredDistance = 1 + u*u + v*v;
            weights[std::size_t(y) * resolution + x] =
                (topFace ? real_t(1) : v) * step*step / (GEOM_PI * squaredDistance*squaredDistance);
        }
    }
    return weights;
}

template<class Worker>
void runProjectionWorkers(std::size_t workerCount, const Worker& work)
{
    if (workerCount == 1) {
        work();
        return;
    }
    std::vector<std::future<void>> workers;
    workers.reserve(workerCount - 1);
    boost::asio::thread_pool pool(workerCount - 1);
    for (std::size_t i = 1; i < workerCount; ++i) {
        std::packaged_task<void()> task(work);
        workers.push_back(task.get_future());
        boost::asio::post(pool, [task = std::move(task)]() mutable { task(); });
    }
    work();
    pool.join();
    for (auto& worker : workers) worker.get();
}

void validateProjectionMesh(const Point3ArrayPtr& points, const Index3ArrayPtr& triangles)
{
    if (!points || !triangles)
        throw std::invalid_argument("Projection requires points and triangle indices");
    if (triangles->size() >= Shape::NOID)
        throw std::invalid_argument("Too many triangles for projection identifiers");
    for (const auto& triangle : *triangles) {
        if (triangle[0] >= points->size() || triangle[1] >= points->size() ||
            triangle[2] >= points->size())
            throw std::invalid_argument("Invalid projection triangle vertex index");
    }
}

void validateHemicubeMesh(const Point3ArrayPtr& points, const Index3ArrayPtr& triangles,
                          uint16_t resolution, real_t nearPlane, real_t farPlane)
{
    validateProjectionMesh(points, triangles);
    if (resolution < 2 || resolution % 2 != 0)
        throw std::invalid_argument("Hemicube resolution must be a positive even integer");
    if (!std::isfinite(nearPlane) || !std::isfinite(farPlane) ||
        nearPlane <= 0 || farPlane <= nearPlane)
        throw std::invalid_argument("Hemicube clipping planes must satisfy 0 < near < far");

}

void validateHemicubeEmitter(const Vector3& center, const Vector3& normal,
                             uint32_t emitterId, std::size_t triangleCount)
{
    if (!center.isValid() || !normal.isValid() || norm(normal) <= GEOM_EPSILON)
        throw std::invalid_argument("Hemicube requires a finite center and nonzero normal");
    if (emitterId != Shape::NOID && emitterId >= triangleCount)
        throw std::invalid_argument("Invalid hemicube emitter index");
}

struct HemicubeWeights {
    explicit HemicubeWeights(uint16_t resolution) :
        top(hemicubePixelWeights(resolution, true)),
        side(hemicubePixelWeights(resolution, false))
    {
        // Preserve the summation order of the five rendered faces.
        for (real_t weight : top) total += weight;
        for (int face = 0; face < 4; ++face)
            for (real_t weight : side) total += weight;
    }

    std::vector<real_t> top, side;
    real_t total = 0;
};

class HemicubeWorkspace {
public:
    HemicubeWorkspace(std::size_t pointCount, std::size_t triangleCount,
                      uint16_t resolution, real_t nearPlane, real_t farPlane,
                      const HemicubeWeights& weights) :
        top(resolution, resolution, ZBufferEngine::eIdBased, Color3::BLACK, Shape::NOID, false),
        side(resolution, resolution / 2, ZBufferEngine::eIdBased, Color3::BLACK, Shape::NOID, false),
        weights(weights),
        cameraPoints(pointCount), contributions(triangleCount, 0),
        nearPlane(nearPlane), farPlane(farPlane)
    {
        configureHemicubeFace(top, true, nearPlane, farPlane);
        configureHemicubeFace(side, false, nearPlane, farPlane);
        polygon.reserve(9);
        clipped.reserve(9);
    }

    HemicubeFormFactorRow compute(const Point3Array& points, const Index3Array& triangles,
                                  const Vector3& center, const Vector3& normal, uint32_t emitterId)
    {
        const Vector3 n = normal.normed();
        const Vector3 right = cross(n, n.anOrthogonalVector()).normed();
        const Vector3 up = cross(right, n);
        const std::array<Vector3, 5> directions = {n, right, -right, up, -up};
        real_t escaped = 0;
        for (std::size_t face = 0; face < directions.size(); ++face) {
            const bool topFace = face == 0;
            auto& buffer = topFace ? top : side;
            const auto& faceWeights = topFace ? weights.top : weights.side;
            const auto depth = buffer.getDepthBuffer();
            const auto ids = buffer.getIdBuffer();
            std::fill(depth->begin(), depth->end(), REAL_MAX);
            std::fill(ids->begin(), ids->end(), Shape::NOID);
            renderHemicubeFace(buffer, points, triangles, center, directions[face], topFace ? up : n,
                               topFace, emitterId, nearPlane, farPlane, cameraPoints, polygon, clipped);
            auto weight = faceWeights.begin();
            for (uint16_t y = 0; y < buffer.getImageHeight(); ++y) {
                for (uint16_t x = 0; x < buffer.getImageWidth(); ++x, ++weight) {
                    const uint32_t id = ids->getAt(x, y);
                    if (id == Shape::NOID) escaped += *weight;
                    else contributions[id] += *weight;
                }
            }
        }

        HemicubeFormFactorRow result;
        for (uint32_t id = 0; id < contributions.size(); ++id) {
            if (contributions[id] == 0) continue;
            result.first.emplace(id, contributions[id] / weights.total);
            contributions[id] = 0;
        }
        result.second = escaped / weights.total;
        return result;
    }

private:
    ZBufferEngine top, side;
    const HemicubeWeights& weights;
    Point3Array cameraPoints;
    std::vector<Vector3> polygon, clipped;
    std::vector<real_t> contributions;
    real_t nearPlane, farPlane;
};

}

HemicubeFormFactorRow PGL(hemicubeFormFactors)(
    const Point3ArrayPtr& points, const Index3ArrayPtr& triangles,
    const Vector3& center, const Vector3& normal, uint16_t resolution,
    uint32_t emitterId, real_t nearPlane, real_t farPlane)
{
    validateHemicubeMesh(points, triangles, resolution, nearPlane, farPlane);
    validateHemicubeEmitter(center, normal, emitterId, triangles->size());
    const HemicubeWeights weights(resolution);
    HemicubeWorkspace workspace(points->size(), triangles->size(), resolution, nearPlane, farPlane, weights);
    return workspace.compute(*points, *triangles, center, normal, emitterId);
}

std::vector<HemicubeFormFactorRow> PGL(hemicubeFormFactorsBatch)(
    const Point3ArrayPtr& points, const Index3ArrayPtr& triangles,
    const Point3ArrayPtr& centers, const Point3ArrayPtr& normals, uint16_t resolution,
    const Index& emitterIds, real_t nearPlane, real_t farPlane, unsigned threads)
{
    if (threads == 0)
        throw std::invalid_argument("Hemicube batch requires at least one thread");
    validateHemicubeMesh(points, triangles, resolution, nearPlane, farPlane);
    if (!centers || !normals || centers->size() != normals->size() ||
        (!emitterIds.empty() && emitterIds.size() != centers->size()))
        throw std::invalid_argument("Hemicube batch requires one normal and optional triangle ID per center");

    const std::size_t count = centers->size();
    if (count == 0) return {};
    const auto emitterId = [&](std::size_t i) {
        return emitterIds.empty() ? Shape::NOID : emitterIds[i];
    };
    for (std::size_t i = 0; i < count; ++i) {
        validateHemicubeEmitter(centers->getAt(i), normals->getAt(i), emitterId(i), triangles->size());
    }

    const HemicubeWeights weights(resolution);
    std::vector<HemicubeFormFactorRow> rows(count);
    const auto computeRow = [&](HemicubeWorkspace& workspace, std::size_t i) {
        rows[i] = workspace.compute(*points, *triangles, centers->getAt(i), normals->getAt(i), emitterId(i));
    };
    const std::size_t workerCount = std::min<std::size_t>(threads, count);
    if (workerCount == 1) {
        HemicubeWorkspace workspace(points->size(), triangles->size(), resolution, nearPlane, farPlane, weights);
        for (std::size_t i = 0; i < count; ++i) computeRow(workspace, i);
        return rows;
    }

    std::atomic<std::size_t> next{0};
    const auto work = [&]() {
        // Each worker owns its buffers and writes disjoint result slots.
        HemicubeWorkspace workspace(points->size(), triangles->size(), resolution, nearPlane, farPlane, weights);
        for (auto i = next.fetch_add(1, std::memory_order_relaxed); i < count;
             i = next.fetch_add(1, std::memory_order_relaxed))
            computeRow(workspace, i);
    };
    runProjectionWorkers(workerCount, work);
    return rows;
}

RealArrayPtr PGL(directionalFacePowersBatch)(
    const Point3ArrayPtr& points, const Index3ArrayPtr& triangles, const Point3ArrayPtr& normals,
    const Point3ArrayPtr& directions, const RealArrayPtr& irradiances, real_t pixelSize, unsigned threads)
{
    validateProjectionMesh(points, triangles);
    if (!normals || normals->size() != triangles->size() ||
        !directions || !irradiances || directions->size() != irradiances->size())
        throw std::invalid_argument("Projection requires a normal per triangle and an irradiance per direction");
    if (threads == 0 || !std::isfinite(pixelSize) || pixelSize < 0)
        throw std::invalid_argument("Projection requires positive thread count and nonnegative pixel size");
    const std::size_t triangleCount = triangles->size(), directionCount = directions->size();
    RealArrayPtr powers(new RealArray(2 * triangleCount, 0));
    if (triangleCount == 0 || directionCount == 0) return powers;

    // Match the bounding box of the rasterized geometry, ignoring unused vertices.
    Vector3 low = points->getAt(triangles->getAt(0)[0]), high = low;
    for (const auto& triangle : *triangles)
        for (int i = 0; i < 3; ++i) {
            const auto& point = points->getAt(triangle[i]);
            for (int axis = 0; axis < 3; ++axis) {
                low[axis] = std::min(low[axis], point[axis]);
                high[axis] = std::max(high[axis], point[axis]);
            }
        }
    const Vector3 center = (low + high) / 2;
    const Vector3 extent = high - low;
    const real_t scale = max3(extent.x(), extent.y(), extent.z());
    if (scale <= 0) return powers;
    if (pixelSize == 0) pixelSize = scale / 100;
    const real_t pixelArea = pixelSize * pixelSize;

    using Contribution = std::pair<std::size_t, real_t>;
    std::vector<std::vector<Contribution>> projected(directionCount);
    const auto project = [&](std::size_t direction, std::vector<uint32_t>& counts) {
        const real_t irradiance = irradiances->getAt(direction);
        Vector3 ray = directions->getAt(direction);
        const real_t length = norm(ray);
        if (!std::isfinite(length) || length == 0 || !std::isfinite(irradiance) || irradiance < 0)
            throw std::invalid_argument("Projection requires finite nonzero directions and nonnegative irradiance");
        if (irradiance == 0) return;
        ray /= length;
        if (std::none_of(normals->begin(), normals->end(), [&](const Vector3& n) { return dot(n, ray) != 0; }))
            return;

        // Use the same camera framing as directionalInterception_zbuffer_from_dir_vectors.
        Vector3 forward = ray;
        Vector3 up = forward.anOrthogonalVector();
        forward.normalize();
        up.normalize();
        Vector3 right = cross(up, forward);
        right.normalize();
        Vector3 projectedUp = cross(forward, right);
        projectedUp.normalize();
        const Matrix3 projection = Matrix3(right, projectedUp, forward).inverse();
        Vector3 projectedLow = projection * low, projectedHigh = projectedLow;
        for (int corner = 0; corner < 8; ++corner) {
            const Vector3 point = projection * Vector3(corner & 1 ? high.x() : low.x(),
                                                     corner & 2 ? high.y() : low.y(),
                                                     corner & 4 ? high.z() : low.z());
            for (int axis = 0; axis < 2; ++axis) {
                projectedLow[axis] = std::min(projectedLow[axis], point[axis]);
                projectedHigh[axis] = std::max(projectedHigh[axis], point[axis]);
            }
        }

        const auto dimension = [&](real_t span) {
            const real_t pixels = std::max(real_t(2), std::ceil(span / pixelSize) + 1);
            if (!std::isfinite(pixels) || pixels > std::numeric_limits<uint16_t>::max())
                throw std::invalid_argument("Directional projection exceeds the raster size limit");
            return static_cast<uint16_t>(pixels);
        };
        const uint16_t width = dimension(projectedHigh.x() - projectedLow.x());
        const uint16_t height = dimension(projectedHigh.y() - projectedLow.y());
        ZBufferEngine buffer(width, height, ZBufferEngine::eIdBased, Color3::BLACK, Shape::NOID, false);
        const real_t halfWidth = width * pixelSize / 2, halfHeight = height * pixelSize / 2;
        buffer.setOrthographicCamera(-halfWidth, halfWidth, -halfHeight, halfHeight, scale, 3 * scale);
        buffer.lookAt(center - forward * scale * 2, center, up);
        const auto camera = buffer.camera();
        for (uint32_t id = 0; id < triangleCount; ++id) {
            const auto& triangle = triangles->getAt(id);
            buffer.renderShadedTriangle(points->getAt(triangle[0]), points->getAt(triangle[1]),
                                        points->getAt(triangle[2]), true, id, TriangleShaderPtr(), camera);
        }

        std::fill(counts.begin(), counts.end(), 0);
        const auto ids = buffer.getIdBuffer();
        for (uint32_t id : *ids)
            if (id != Shape::NOID) ++counts[id];
        auto& result = projected[direction];
        for (std::size_t id = 0; id < triangleCount; ++id) {
            if (counts[id] == 0) continue;
            const real_t cosine = -dot(normals->getAt(id), ray);
            if (cosine != 0)
                result.emplace_back(2 * id + (cosine < 0), counts[id] * pixelArea * irradiance);
        }
    };

    std::atomic<std::size_t> next{0};
    const auto work = [&]() {
        std::vector<uint32_t> counts(triangleCount);
        for (auto i = next.fetch_add(1, std::memory_order_relaxed); i < directionCount;
             i = next.fetch_add(1, std::memory_order_relaxed))
            project(i, counts);
    };
    runProjectionWorkers(std::min<std::size_t>(threads, directionCount), work);
    // Keep the same accumulation order for all thread counts.
    for (const auto& direction : projected)
        for (const auto& contribution : direction)
            (*powers)[contribution.first] += contribution.second;
    return powers;
}

namespace {

// Compensated summation keeps the stopping criterion stable for small residuals.
double sumRadiosityPower(const std::vector<double>& powers)
{
    double sum = 0, correction = 0;
    for (double power : powers) {
        const double next = sum + power;
        correction += sum >= power ? (sum - next) + power : (power - next) + sum;
        sum = next;
    }
    return sum + correction;
}

}

struct RadiosityTransport::Impl {
    struct Link {
        std::size_t face;
        double factor, rho, tau, absorptance;
    };

    struct Row {
        double escape;
        std::vector<Link> links;
    };

    Point3ArrayPtr points;
    Index3ArrayPtr triangles;
    Point3ArrayPtr centers, normals;
    std::vector<double> reflectance, transmittance;
    bool twoSided;
    uint16_t resolution;
    real_t nearPlane, farPlane;
    unsigned threads;
    std::vector<std::optional<Row>> rows;

    void prepareRows(const std::vector<double>& pending)
    {
        std::vector<std::size_t> missing;
        for (std::size_t face = 0; face < pending.size(); ++face)
            if (pending[face] != 0 && !rows[face]) missing.push_back(face);
        if (missing.empty()) return;

        Point3ArrayPtr batchCenters(new Point3Array(missing.size()));
        Point3ArrayPtr batchNormals(new Point3Array(missing.size()));
        Index emitterIds(missing.size());
        for (std::size_t i = 0; i < missing.size(); ++i) {
            const std::size_t face = missing[i], source = face / 2;
            batchCenters->setAt(i, centers->getAt(source));
            const Vector3& normal = normals->getAt(source);
            batchNormals->setAt(i, face % 2 ? -normal : normal);
            emitterIds[i] = static_cast<uint32_t>(source);
        }
        const auto factors = hemicubeFormFactorsBatch(points, triangles, batchCenters, batchNormals,
                                                      resolution, emitterIds, nearPlane, farPlane, threads);
        for (std::size_t i = 0; i < missing.size(); ++i) {
            const std::size_t face = missing[i], source = face / 2;
            Row row{factors[i].second, {}};
            row.links.reserve(factors[i].first.size());
            const auto& origin = centers->getAt(source);
            for (const auto& entry : factors[i].first) {
                const std::size_t target = entry.first;
                const bool front = dot(normals->getAt(target), origin - centers->getAt(target)) > 0;
                const double rho = front || twoSided ? reflectance[target] : 0.0;
                const double tau = transmittance[target];
                row.links.push_back({2 * target + !front, entry.second, rho, tau, 1 - (rho + tau)});
            }
            // Process links in triangle order, as in the Python solver.
            std::sort(row.links.begin(), row.links.end(),
                      [](const Link& a, const Link& b) { return a.face < b.face; });
            rows[face] = std::move(row);
        }
    }
};

RadiosityTransport::RadiosityTransport(
    const Point3ArrayPtr& points, const Index3ArrayPtr& triangles,
    const Point3ArrayPtr& centers, const Point3ArrayPtr& normals,
    const std::vector<double>& reflectance, const std::vector<double>& transmittance, bool twoSided,
    uint16_t resolution, real_t nearPlane, real_t farPlane, unsigned threads)
{
    validateHemicubeMesh(points, triangles, resolution, nearPlane, farPlane);
    const std::size_t count = triangles->size();
    if (!centers || !normals || centers->size() != count || normals->size() != count ||
        reflectance.size() != count || transmittance.size() != count || threads == 0)
        throw std::invalid_argument("Radiosity transport requires geometry and optics for every triangle");
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(reflectance[i]) || !std::isfinite(transmittance[i]) ||
            reflectance[i] < 0 || transmittance[i] < 0 || reflectance[i] + transmittance[i] > 1)
            throw std::invalid_argument("Invalid radiosity transport coefficients");
    }
    data.reset(new Impl{points, triangles, centers, normals, reflectance, transmittance, twoSided,
                        resolution, nearPlane, farPlane, threads,
                        std::vector<std::optional<Impl::Row>>(2 * count)});
}

RadiosityTransport::~RadiosityTransport() = default;

RadiosityPropagation RadiosityTransport::propagate(
    std::vector<double> pending, std::size_t maxBounces, double threshold)
{
    if (pending.size() != data->rows.size() || !std::isfinite(threshold) || threshold < 0 ||
        std::any_of(pending.begin(), pending.end(), [](double p) { return !std::isfinite(p) || p < 0; }))
        throw std::invalid_argument("Invalid power supplied to radiosity transport");

    const auto count = data->triangles->size();
    RadiosityPropagation result;
    result.outgoing = pending;
    result.incident.resize(count, 0);
    result.absorbed.resize(count, 0);
    result.remaining = sumRadiosityPower(pending);
    std::vector<double> next(pending.size());
    while (result.bounces < maxBounces && result.remaining > threshold) {
        data->prepareRows(pending);
        std::fill(next.begin(), next.end(), 0);
        for (std::size_t source = 0; source < pending.size(); ++source) {
            const double power = pending[source];
            if (power == 0) continue;
            const auto& row = *data->rows[source];
            const double escaped = power * row.escape;
            result.escaped += escaped;
            for (const auto& link : row.links) {
                const std::size_t face = link.face, target = face / 2;
                const std::size_t opposite = face ^ 1;
                const double received = power * link.factor;
                const double reflected = received * link.rho;
                const double transmitted = received * link.tau;
                const double absorbed = received * link.absorptance;
                result.incident[target] += received;
                result.absorbed[target] += absorbed;
                next[face] += reflected;
                next[opposite] += transmitted;
                result.outgoing[face] += reflected;
                result.outgoing[opposite] += transmitted;
            }
        }
        pending.swap(next);
        result.remaining = sumRadiosityPower(pending);
        ++result.bounces;
    }
    return result;
}
