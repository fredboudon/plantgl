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


#ifndef __pointmanipulation_h__
#define __pointmanipulation_h__

#include "../algo_config.h"
#include <plantgl/math/util_math.h>
#include <plantgl/math/util_matrix.h>
#include <plantgl/tool/rcobject.h>
#include <plantgl/algo/grid/regularpointgrid.h>
#include <plantgl/scenegraph/container/indexarray.h>
#include <plantgl/scenegraph/function/function.h>
#include <plantgl/scenegraph/scene/scene.h>
#include <plantgl/scenegraph/geometry/pointset.h>
#include <plantgl/tool/util_array.h>
#include <plantgl/tool/util_array2.h>
#include <plantgl/math/util_vector.h>
#include <memory>
#include <vector>

PGL_BEGIN_NAMESPACE


  template<class PointListType>
  RCPtr<PointListType> contract_point(RCPtr<PointListType> points, real_t radius) {
    typedef typename PointListType::element_type VectorType;
    typedef PointRefGrid<PointListType> LocalPointGrid;
    typedef typename LocalPointGrid::PointIndexList PointIndexList;

    LocalPointGrid grid(radius, points);

    RCPtr<PointListType> result(new PointListType(points->size()));
    typename PointListType::iterator _itresult = result->begin();
    for (typename PointListType::const_iterator _itsource = points->begin();
         _itsource != points->end(); ++_itsource, ++_itresult) {
      PointIndexList pointindices = grid.query_ball_point(*_itsource, radius);
      VectorType center;
      if (pointindices.size() > 0) {
        for (typename PointIndexList::const_iterator itptindex = pointindices.begin();
             itptindex != pointindices.end(); ++itptindex) { center += points->getAt(*itptindex); }
        center /= pointindices.size();
        *_itresult = center;
      } else *_itresult = *_itsource;

    }

    return result;
  }

  ALGO_API Color4ArrayPtr generate_point_color(PointSet &point);

  ALGO_API Index
  select_soil(const Point3ArrayPtr &point, IndexArrayPtr &kclosest, const uint_t &topHeightPourcent, const real_t &bottomThreshold);

  ALGO_API std::pair<uint_t, uint_t> find_min_max(const Point3ArrayPtr &point, const uint_t &boundMaxPourcent);

  ALGO_API std::pair<uint_t, uint_t>
  find_min_max(const Point3ArrayPtr &point, const uint_t &boundPourcent, const Vector3 &center,
               const Vector3 &direction);

  ALGO_API Index get_shortest_path(const Point3ArrayPtr &point, IndexArrayPtr &kclosest, const uint_t &point_begin,
                                   const uint_t &point_end);

  ALGO_API std::pair<Point3ArrayPtr, Index>
  add_baricenter_points_of_path(const Point3ArrayPtr &point, IndexArrayPtr &kclosest, const Index &path,
                                const real_t &radius);

  ALGO_API RealArrayPtr get_radii_of_path(const Point3ArrayPtr &point,
                                                 const IndexArrayPtr &kclosest,
                                                 const Index &path,
                                                 const real_t &around_radius);

  ALGO_API real_t
  get_average_radius_of_path(const Point3ArrayPtr &point, const IndexArrayPtr &kclosest, const Index &path);

  ALGO_API Index
  select_point_around_line(const Point3ArrayPtr &point, const Vector3 &center, const Vector3 &direction,
                           const real_t &radius);

  ALGO_API Index
  select_wire_from_path(const Point3ArrayPtr &point, const Index &path, const real_t &radius, const RealArrayPtr &radii);

  ALGO_API Index
  select_r_isolate_points(const IndexArrayPtr &rneighborhoods, const real_t &radius, const real_t &mindensity);

  ALGO_API Index select_k_isolate_points(const Point3ArrayPtr &point, const IndexArrayPtr &kclosest, const uint32_t &k,
                                         const real_t &mindensity);

  ALGO_API Index filter_min_densities(const RealArrayPtr densities, const real_t &densityratio);
  ALGO_API Index filter_max_densities(const RealArrayPtr densities, const real_t &densityratio);

  ALGO_API std::pair<Index, real_t>
  select_pole_from_point(const Point3ArrayPtr &points, const Vector3 &startPoint, std::size_t iterations, real_t maxAngle);
  
  ALGO_API std::pair<Index, real_t>
  select_pole_points(const Point3ArrayPtr &point, real_t radius, uint_t iterations, real_t tolerance = -1.0);

  ALGO_API std::pair<Index, real_t>
  select_pole_points_mt(const Point3ArrayPtr &point, real_t radius, uint_t iterations, real_t tolerance = -1.0);

// typedef std::vector<std::vector<uint32_t> > AdjacencyMap;

/// K-Neighborhood computation
  ALGO_API IndexArrayPtr
  delaunay_point_connection(const Point3ArrayPtr points);

  ALGO_API Index3ArrayPtr
  delaunay_triangulation(const Point3ArrayPtr points);

  ALGO_API IndexArrayPtr
  k_closest_points_from_delaunay(const Point3ArrayPtr points, size_t k);

  ALGO_API IndexArrayPtr
  k_closest_points_from_ann(const Point3ArrayPtr points, size_t k, bool symmetric = false);

// ALGO_API IndexArrayPtr
// k_closest_points_from_cgal(const Point3ArrayPtr points, size_t k);

  ALGO_API IndexArrayPtr
  symmetrize_connections(const IndexArrayPtr adjacencies);

  ALGO_API IndexArrayPtr
  get_all_connex_components(const Point3ArrayPtr points, const IndexArrayPtr adjacencies, bool verbose = false);

/// Reconnect all connex components of an adjacency graph
  ALGO_API IndexArrayPtr
  connect_all_connex_components(const Point3ArrayPtr points, const IndexArrayPtr adjacencies, bool verbose = false);

/// R-Neighborhood computation
  ALGO_API Index
  r_neighborhood(uint32_t pid, const Point3ArrayPtr &points, const IndexArrayPtr &adjacencies, const real_t radius);

  ALGO_API IndexArrayPtr
  r_neighborhoods(const Point3ArrayPtr points, const IndexArrayPtr adjacencies, const RealArrayPtr radii);

  ALGO_API IndexArrayPtr
  r_neighborhoods(const Point3ArrayPtr points, const IndexArrayPtr adjacencies, real_t radius, bool verbose = false);

  ALGO_API IndexArrayPtr
  r_neighborhoods_mt(const Point3ArrayPtr points, const IndexArrayPtr adjacencies, real_t radius, bool verbose = false);

  ALGO_API Index
  r_anisotropic_neighborhood(uint32_t pid, const Point3ArrayPtr points,
                             const IndexArrayPtr adjacencies,
                             const real_t radius,
                             const Vector3 &direction,
                             const real_t alpha, const real_t beta);

  ALGO_API IndexArrayPtr
  r_anisotropic_neighborhoods(const Point3ArrayPtr points,
                              const IndexArrayPtr adjacencies,
                              const RealArrayPtr radii,
                              const Point3ArrayPtr directions,
                              const real_t alpha,
                              const real_t beta);

  ALGO_API IndexArrayPtr
  r_anisotropic_neighborhoods(const Point3ArrayPtr points,
                              const IndexArrayPtr adjacencies,
                              const real_t radius,
                              const Point3ArrayPtr directions,
                              const real_t alpha,
                              const real_t beta);

/// Extended K-Neighborhood computation
  ALGO_API Index
  k_neighborhood(uint32_t pid, const Point3ArrayPtr points, const IndexArrayPtr adjacencies, const uint32_t k);

  ALGO_API IndexArrayPtr
  k_neighborhoods(const Point3ArrayPtr points, const IndexArrayPtr adjacencies, const uint32_t k);


// Useful function

/// Find the k closest point from the set of adjacencies
  ALGO_API Index
  get_k_closest_from_n(const Index &adjacencies, const uint32_t k, uint32_t pid, const Point3ArrayPtr points);


  ALGO_API real_t
  pointset_max_distance(uint32_t pid,
                        const Point3ArrayPtr points,
                        const Index &group);

  ALGO_API real_t
  pointset_max_distance(const Vector3 &origin,
                        const Point3ArrayPtr points,
                        const Index &group);

  ALGO_API real_t
  pointset_min_distance(uint32_t pid,
                        const Point3ArrayPtr points,
                        const Index &group);

  ALGO_API real_t
  pointset_min_distance(const Vector3 &origin,
                        const Point3ArrayPtr points,
                        const Index &group);

// ALGO_API
  template<class IndexGroup>
  real_t pointset_mean_distance(const Vector3 &origin,
                                const Point3ArrayPtr points,
                                const IndexGroup &group) {
    if (group.empty()) return 0;
    real_t sum_distance = 0;
    for (typename IndexGroup::const_iterator it = group.begin(); it != group.end(); ++it)
      sum_distance += norm(origin - points->getAt(*it));
    return sum_distance / group.size();
  }

  template<class IndexGroupArray>
  RealArrayPtr pointset_mean_distances(const Point3ArrayPtr origins,
                                              const Point3ArrayPtr points,
                                              const RCPtr<IndexGroupArray> groups) {
    typedef typename IndexGroupArray::element_type IndexGroup;
    RealArrayPtr result(new RealArray(groups->size()));
    RealArray::iterator itres = result->begin();
    Point3Array::const_iterator itorigin = origins->begin();
    for (typename IndexGroupArray::const_iterator it = groups->begin(); it != groups->end(); ++it, ++itorigin, ++itres)
      *itres = pointset_mean_distance<IndexGroup>(*itorigin, points, *it);
    return result;
  }

  ALGO_API real_t
  pointset_mean_radial_distance(const Vector3 &origin,
                                const Vector3 &direction,
                                const Point3ArrayPtr points,
                                const Index &group);

  ALGO_API real_t
  pointset_max_radial_distance(const Vector3 &origin,
                               const Vector3 &direction,
                               const Point3ArrayPtr points,
                               const Index &group);


  ALGO_API Matrix3 pointset_covariance(const Point3ArrayPtr points, const Index &group = Index());


  ALGO_API Index
  get_sorted_element_order(const RealArrayPtr distances);


/// Density computation
  ALGO_API real_t
  density_from_r_neighborhood(uint32_t pid,
                              const Point3ArrayPtr points,
                              const IndexArrayPtr adjacencies,
                              const real_t radius);

  ALGO_API RealArrayPtr
  densities_from_r_neighborhood(const Point3ArrayPtr points,
                                const IndexArrayPtr adjacencies,
                                const real_t radius);

  ALGO_API RealArrayPtr
  densities_from_r_neighborhood(const IndexArrayPtr neighborhood,
                                const real_t radius);


// if k == 0, then k is directly the nb of point given in adjacencies.
  ALGO_API real_t
  density_from_k_neighborhood(uint32_t pid,
                              const Point3ArrayPtr points,
                              const IndexArrayPtr adjacencies,
                              const uint32_t k = 0);

  ALGO_API RealArrayPtr
  densities_from_k_neighborhood(const Point3ArrayPtr points,
                                const IndexArrayPtr adjacencies,
                                const uint32_t k = 0);



/// Orientation estimations

  ALGO_API std::pair<Vector3, Vector3>
  pointset_plane(const Point3ArrayPtr points, const Index &group);

  ALGO_API Vector3
  pointset_orientation(const Point3ArrayPtr points, const Index &group);

  ALGO_API Point3ArrayPtr
  pointsets_orientations(const Point3ArrayPtr points, const IndexArrayPtr groups);

  ALGO_API Vector3
  pointset_normal(const Point3ArrayPtr points, const Index &group);

  ALGO_API Point3ArrayPtr
  pointsets_normals(const Point3ArrayPtr points, const IndexArrayPtr groups);


  ALGO_API Point3ArrayPtr
  pointsets_orient_normals(const Point3ArrayPtr normals, const Point3ArrayPtr points, const IndexArrayPtr riemanian);


  ALGO_API Point3ArrayPtr
  pointsets_orient_normals(const Point3ArrayPtr normals, uint32_t source, const IndexArrayPtr riemanian);

/// Orientation estimations
  ALGO_API Vector3
  triangleset_orientation(const Point3ArrayPtr points, const Index3ArrayPtr triangles);


  struct CurvatureInfo {
    Vector3 origin;
    Vector3 maximal_principal_direction;
    real_t maximal_curvature;
    Vector3 minimal_principal_direction;
    real_t minimal_curvature;
    Vector3 normal;
  };

  ALGO_API CurvatureInfo
  principal_curvatures(const Point3ArrayPtr points, uint32_t pid, const Index &group, size_t fitting_degree = 4,
                       size_t monge_degree = 4);

  ALGO_API std::vector<CurvatureInfo>
  principal_curvatures(const Point3ArrayPtr points, const IndexArrayPtr groups, size_t fitting_degree = 4,
                       size_t monge_degree = 4);

  ALGO_API std::vector<CurvatureInfo>
  principal_curvatures(const Point3ArrayPtr points, const IndexArrayPtr adjacencies, real_t radius,
                       size_t fitting_degree = 4, size_t monge_degree = 4);

// Compute the set of points that are at a distance < width from the plane at point pid in direction
  ALGO_API Index
  point_section(uint32_t pid,
                const Point3ArrayPtr points,
                const IndexArrayPtr adjacencies,
                const Vector3 &direction,
                real_t width);

  ALGO_API Index
  point_section(uint32_t pid,
                const Point3ArrayPtr points,
                const IndexArrayPtr adjacencies,
                const Vector3 &direction,
                real_t width,
                real_t maxradius);

  ALGO_API IndexArrayPtr
  points_sections(const Point3ArrayPtr points,
                  const IndexArrayPtr adjacencies,
                  const Point3ArrayPtr directions,
                  real_t width);

/// Compute a circle from a point set
  ALGO_API std::pair<Vector3, real_t>
  pointset_circle(const Point3ArrayPtr points,
                  const Index &group,
                  bool bounding = false);

  ALGO_API std::pair<Vector3, real_t>
  pointset_circle(const Point3ArrayPtr points,
                  const Index &group,
                  const Vector3 &direction,
                  bool bounding = false);

  ALGO_API std::pair<Point3ArrayPtr, RealArrayPtr>
  pointsets_circles(const Point3ArrayPtr points,
                    const IndexArrayPtr groups,
                    const Point3ArrayPtr directions = Point3ArrayPtr(0),
                    bool bounding = false);

  ALGO_API std::pair<Point3ArrayPtr, RealArrayPtr>
  pointsets_section_circles(const Point3ArrayPtr points,
                            const IndexArrayPtr adjacencies,
                            const Point3ArrayPtr directions,
                            real_t width,
                            bool bounding = false);


// Adaptive contraction
  ALGO_API std::pair<Point3ArrayPtr, RealArrayPtr>
  adaptive_section_circles(const Point3ArrayPtr points,
                           const IndexArrayPtr adjacencies,
                           const Point3ArrayPtr orientations,
                           const RealArrayPtr widths,
                           const RealArrayPtr maxradii);

// Adaptive contraction
  ALGO_API std::pair<Point3ArrayPtr, RealArrayPtr>
  adaptive_section_circles(const Point3ArrayPtr points,
                           const IndexArrayPtr adjacencies,
                           const Point3ArrayPtr orientations,
                           const real_t width,
                           const RealArrayPtr maxradii);

// adaptive contraction
  ALGO_API RealArrayPtr
  adaptive_radii(const RealArrayPtr density,
                 real_t minradius, real_t maxradius,
                 QuantisedFunctionPtr densityradiusmap = NULL);

// Adaptive contraction
  ALGO_API Point3ArrayPtr
  adaptive_contration(const Point3ArrayPtr points,
                      const Point3ArrayPtr orientations,
                      const IndexArrayPtr adjacencies,
                      const RealArrayPtr densities,
                      real_t minradius, real_t maxradius,
                      QuantisedFunctionPtr densityradiusmap = NULL,
                      const real_t alpha = 1,
                      const real_t beta = 1);

// Adaptive contraction
  ALGO_API std::pair<Point3ArrayPtr, RealArrayPtr>
  adaptive_section_contration(const Point3ArrayPtr points,
                              const Point3ArrayPtr orientations,
                              const IndexArrayPtr adjacencies,
                              const RealArrayPtr densities,
                              real_t minradius, real_t maxradius,
                              QuantisedFunctionPtr densityradiusmap = NULL,
                              const real_t alpha = 1,
                              const real_t beta = 1);

/// Shortest path
  ALGO_API std::pair<Uint32Array1Ptr, RealArrayPtr>
  points_dijkstra_shortest_path(const Point3ArrayPtr points,
                                const IndexArrayPtr adjacencies,
                                uint32_t root,
                                real_t powerdist = 1);


// Return groups of points
  ALGO_API IndexArrayPtr
  quotient_points_from_adjacency_graph(const real_t binsize,
                                       const Point3ArrayPtr points,
                                       const IndexArrayPtr adjacencies,
                                       const RealArrayPtr distances_to_root);

// Return adjacencies between groups
  ALGO_API IndexArrayPtr
  quotient_adjacency_graph(const IndexArrayPtr adjacencies,
                           const IndexArrayPtr groups);

  ALGO_API Vector3
  centroid_of_group(const Point3ArrayPtr points,
                    const Index &group);

  ALGO_API Point3ArrayPtr
  centroids_of_groups(const Point3ArrayPtr points,
                      const IndexArrayPtr groups);


  template<class IndexGroup>
  Vector3 centroid_of_group(const Point3ArrayPtr points,
                                   const IndexGroup &group) {
    Vector3 gcentroid;
    real_t nbpoints = 0;
    for (typename IndexGroup::const_iterator itn = group.begin(); itn != group.end(); ++itn, ++nbpoints) {
      gcentroid += points->getAt(*itn);
    }
    return gcentroid / nbpoints;
  }


  template<class IndexGroupArray>
  Point3ArrayPtr centroids_of_groups(const Point3ArrayPtr points,
                                     const RCPtr<IndexGroupArray> groups) {
    Point3ArrayPtr result(new Point3Array(groups->size()));
    uint32_t cgroup = 0;
    for (typename IndexGroupArray::const_iterator itgs = groups->begin(); itgs != groups->end(); ++itgs, ++cgroup) {
      result->setAt(cgroup, centroid_of_group(points, *itgs));
    }
    return result;
  }

  ALGO_API IndexArrayPtr cluster_points(const Point3ArrayPtr points, const Point3ArrayPtr clustercentroid);

  ALGO_API Uint32Array1Ptr points_clusters(const Point3ArrayPtr points, const Point3ArrayPtr clustercentroid);

// Xu 07 method for main branching system
  ALGO_API Point3ArrayPtr
  skeleton_from_distance_to_root_clusters(const Point3ArrayPtr points, uint32_t root, real_t binsize, uint32_t k,
                                          Uint32Array1Ptr &group_parents, IndexArrayPtr &group_components,
                                          bool connect_all_points = false, bool verbose = false);

  ALGO_API Index
  points_in_range_from_root(const real_t initialdist, const real_t binsize,
                            const RealArrayPtr distances_to_root);

  ALGO_API std::pair<IndexArrayPtr, RealArrayPtr>
  next_quotient_points_from_adjacency_graph(const real_t initiallevel,
                                            const real_t binsize,
                                            const Index &currents,
                                            const IndexArrayPtr adjacencies,
                                            const RealArrayPtr distances_to_root);



// Livny method procedures
// compute parent-children relation from child-parent relation
  ALGO_API IndexArrayPtr determine_children(const Uint32Array1Ptr parents, uint32_t &root);

// compute a weight to each points as sum of length of carried segments
  ALGO_API RealArrayPtr carried_length(const Point3ArrayPtr points, const Uint32Array1Ptr parents);

// compute a weight to each points as number of node in their
  ALGO_API Uint32Array1Ptr subtrees_size(const Uint32Array1Ptr parents);

  ALGO_API Uint32Array1Ptr subtrees_size(const IndexArrayPtr children, uint32_t root);

// optimize orientation
  ALGO_API Point3ArrayPtr optimize_orientations(const Point3ArrayPtr points,
                                                const Uint32Array1Ptr parents,
                                                const RealArrayPtr weights);

// optimize orientation
  ALGO_API Point3ArrayPtr optimize_positions(const Point3ArrayPtr points,
                                             const Point3ArrayPtr orientations,
                                             const Uint32Array1Ptr parents,
                                             const RealArrayPtr weights);

// estimate average radius around edges
  ALGO_API real_t average_radius(const Point3ArrayPtr points,
                                 const Point3ArrayPtr nodes,
                                 const Uint32Array1Ptr parents,
                                 uint32_t maxclosestnodes = 10);

  ALGO_API RealArrayPtr distance_to_shape(const Point3ArrayPtr points,
                                                 const Point3ArrayPtr nodes,
                                                 const Uint32Array1Ptr parents,
                                                 const RealArrayPtr radii,
                                                 uint32_t maxclosestnodes = 10);

  ALGO_API real_t average_distance_to_shape(const Point3ArrayPtr points,
                                            const Point3ArrayPtr nodes,
                                            const Uint32Array1Ptr parents,
                                            const RealArrayPtr radii,
                                            uint32_t maxclosestnodes = 10);

  ALGO_API Index points_at_distance_from_skeleton(const Point3ArrayPtr points,
                                                  const Point3ArrayPtr nodes,
                                                  const Uint32Array1Ptr parents,
                                                  real_t distance,
                                                  uint32_t maxclosestnodes = 10);

  ALGO_API RealArrayPtr estimate_radii_from_points(const Point3ArrayPtr points,
                                                          const Point3ArrayPtr nodes,
                                                          const Uint32Array1Ptr parents,
                                                          bool maxmethod = false,
                                                          uint32_t maxclosestnodes = 10);
// estimate radius for each node
  ALGO_API RealArrayPtr estimate_radii_from_pipemodel(const Point3ArrayPtr nodes,
                                                             const Uint32Array1Ptr parents,
                                                             const RealArrayPtr weights,
                                                             real_t averageradius,
                                                             real_t pipeexponent = 2.5);

  ALGO_API bool node_continuity_test(const Vector3 &node, real_t noderadius,
                                     const Vector3 &parent, real_t parentradius,
                                     const Vector3 &child, real_t childradius,
                                     real_t overlapfilter = 0.5,
                                     bool verbose = false, ScenePtr *visu = NULL);

  ALGO_API bool node_intersection_test(const Vector3 &root, real_t rootradius,
                                       const Vector3 &p1, real_t radius1,
                                       const Vector3 &p2, real_t radius2,
                                       real_t overlapfilter,
                                       bool verbose = false, ScenePtr *visu = NULL);
// compute the minimum maximum and  mean edge length
  ALGO_API Vector3 min_max_mean_edge_length(const Point3ArrayPtr points, const Uint32Array1Ptr parents);

  ALGO_API Vector3 min_max_mean_edge_length(const Point3ArrayPtr points, const IndexArrayPtr graph);

// determine nodes to filter
  ALGO_API Index detect_short_nodes(const Point3ArrayPtr nodes,
                                    const Uint32Array1Ptr parents,
                                    real_t edgelengthfilter = 0.001);

  ALGO_API void remove_nodes(const Index &toremove,
                             Point3ArrayPtr &nodes,
                             Uint32Array1Ptr &parents,
                             RealArrayPtr &radii);

  ALGO_API inline void remove_nodes(const Index &toremove,
                                    Point3ArrayPtr &nodes,
                                    Uint32Array1Ptr &parents) {
    RealArrayPtr radii(0);
    remove_nodes(toremove, nodes, parents, radii);
  }

// determine nodes to filter
  ALGO_API IndexArrayPtr detect_similar_nodes(const Point3ArrayPtr nodes,
                                              const Uint32Array1Ptr parents,
                                              const RealArrayPtr radii,
                                              const RealArrayPtr weights,
                                              real_t overlapfilter = 0.5);

  ALGO_API void merge_nodes(const IndexArrayPtr tomerge,
                            Point3ArrayPtr &nodes,
                            Uint32Array1Ptr &parents,
                            RealArrayPtr &radii,
                            RealArrayPtr weights);


// determine mean direction of a set of points
  ALGO_API Vector3 pointset_mean_direction(const Vector3 &origin, const Point3ArrayPtr points,
                                                  const Index &group = Index());

// determine all directions of a set of points
  ALGO_API Point3ArrayPtr
  pointset_directions(const Vector3 &origin, const Point3ArrayPtr points, const Index &group = Index());

// determine all directions of a set of points
  ALGO_API Point2ArrayPtr
  pointset_angulardirections(const Point3ArrayPtr points, const Vector3 &origin = TOOLS(Vector3::ORIGIN),
                             const Index &group = Index());

// find the closest point from a group
  ALGO_API std::pair<uint32_t, real_t>
  findClosestFromSubset(const Vector3 &origin, const Point3ArrayPtr points, const Index &group = Index());

// compute the pair wise distance between orientation (in angular domain)
  ALGO_API RealArray2Ptr orientations_distances(const Point3ArrayPtr orientations, const Index &group = Index());

// compute the pair wise similarity between orientation (in angular domain)
  ALGO_API RealArray2Ptr orientations_similarities(const Point3ArrayPtr orientations,
                                                          const Index &group = Index());

// compute the points that make the junction of the two group
  ALGO_API std::pair<Index, Index>
  cluster_junction_points(const IndexArrayPtr pointtoppology, const Index &group1, const Index &group2);


// from Tagliasacchi 2009
  ALGO_API Vector3 section_normal(const Point3ArrayPtr pointnormals, const Index &section);

  ALGO_API Point3ArrayPtr sections_normals(const Point3ArrayPtr pointnormals, const IndexArrayPtr &sections);

/*
    Compute the geometric median of a point sample.
    The geometric median coordinates will be expressed in the Spatial Image reference system (not in real world metrics).
    We use the Weiszfeld's algorithm (http://en.wikipedia.org/wiki/Geometric_median)
*/
  ALGO_API uint32_t approx_pointset_median(const Point3ArrayPtr points, uint32_t nbIterMax = 200);

// brute force approach
  ALGO_API uint32_t pointset_median(const Point3ArrayPtr points);

PGL_END_NAMESPACE

#endif
