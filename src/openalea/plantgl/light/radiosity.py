"""Diffuse radiosity on triangle meshes using PlantGL's hemicube."""

from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Real
from operator import index

from .. import algo
from ..algo import _pglalgo
from ..math import Vector3, cross, dot, norm
from ..scenegraph import Index3Array, Point3Array, RealArray, Scene, Shape, TriangleSet
from .light import directionalInterception_from_dir_vectors, eTriangleProjection, eZBufferProjection
from .sky import soc_sky_sources
from .utils import estimate_dir_vectors

__all__ = ["RadiosityResult", "solve_radiosity"]


@dataclass(frozen=True)
class RadiosityResult:
    """Triangle-indexed exitance (W/m²) and incident and absorbed power (W).

    `radiosity` is the sum of the front and back exitances.
    `incident_power` and `absorbed_power` include sun, sky and subsequent bounces.
    `escaped_power` counts diffuse escape after emission or scattering.
    """

    radiosity: tuple[float, ...]
    incident_power: tuple[float, ...]
    absorbed_power: tuple[float, ...]
    emitted_power: float
    escaped_power: float
    remaining_power: float
    bounces: int
    converged: bool
    front_radiosity: tuple[float, ...] = ()
    back_radiosity: tuple[float, ...] = ()
    triangle_area: tuple[float, ...] = ()
    direct_front_power: tuple[float, ...] = ()
    direct_back_power: tuple[float, ...] = ()
    direct_absorbed_power: tuple[float, ...] = ()
    indirect_incident_power: tuple[float, ...] = ()
    indirect_absorbed_power: tuple[float, ...] = ()
    sky_front_power: tuple[float, ...] = ()
    sky_back_power: tuple[float, ...] = ()
    sky_absorbed_power: tuple[float, ...] = ()

    @property
    def solar_intercepted_power(self):
        return fsum(self.direct_front_power) + fsum(self.direct_back_power)

    @property
    def sky_intercepted_power(self):
        return fsum(self.sky_front_power) + fsum(self.sky_back_power)

    @property
    def direct_front_irradiance(self):
        return tuple(power / area for power, area in zip(self.direct_front_power, self.triangle_area))

    @property
    def direct_back_irradiance(self):
        return tuple(power / area for power, area in zip(self.direct_back_power, self.triangle_area))

    @property
    def sky_front_irradiance(self):
        return tuple(power / area for power, area in zip(self.sky_front_power, self.triangle_area))

    @property
    def sky_back_irradiance(self):
        return tuple(power / area for power, area in zip(self.sky_back_power, self.triangle_area))


def _triangle_values(values, count, name, maximum=float("inf")):
    if isinstance(values, Real):
        values = (float(values),) * count
    else:
        values = tuple(map(float, values))
    if len(values) != count or any(not isfinite(v) or not 0 <= v <= maximum for v in values):
        raise ValueError(f"{name} must supply one finite value in [0, {maximum}] per triangle")
    return values


def _external_scene(points, triangles, method, resolution):
    if method not in (eZBufferProjection, eTriangleProjection):
        raise ValueError("solar_method must be eZBufferProjection or eTriangleProjection")
    options = {"multithreaded": False}
    if resolution is not None:
        if method != eZBufferProjection or not isfinite(resolution) or resolution <= 0:
            raise ValueError("solar_resolution must be a positive pixel size for eZBufferProjection")
        options["resolution"] = resolution
    scene = Scene([
        Shape(TriangleSet([points[vertex] for vertex in triangle], [(0, 1, 2)]), id=triangle_id)
        for triangle_id, triangle in enumerate(triangles)
    ])
    return scene, options


def _directional_face_powers(scene, normals, directions, method, options):
    powers = [0.0] * (2 * len(normals))
    # Project each direction separately so its power is assigned to the
    # incident face before combining directions.
    for direction, irradiance in directions:
        ray = Vector3(direction)
        length = norm(ray)
        if not isfinite(length) or length == 0 or not isfinite(irradiance) or irradiance < 0:
            raise ValueError("light directions require a finite nonzero vector and nonnegative irradiance")
        ray /= length
        cosines = [-dot(normal, ray) for normal in normals]
        if irradiance == 0 or not any(cosines):
            continue
        intercepted = directionalInterception_from_dir_vectors(scene, [(ray, irradiance)], method, **options)
        for target, power in intercepted.items():
            cosine = cosines[target]
            if cosine != 0:
                powers[2 * target + int(cosine < 0)] += power
    return powers


def solve_radiosity(
    points, triangles, emission, reflectance, *,
    transmittance=None, resolution=128, max_bounces=50, tolerance=1e-8,
    near=1e-4, far=None, solar_directions=None, solar_method=eZBufferProjection,
    solar_resolution=None, sky_diffuse_irradiance=0, hemicube_threads=1, sky_threads=1,
):
    """Propagate diffuse emission on a triangulated mesh.

    `emission` is emitted power per unit area and `reflectance` is a fraction
    in [0, 1]. Emission starts on the front side given by triangle winding.
    With `transmittance=None`, the back absorbs all incident power. Providing
    `transmittance` enables the same diffuse reflection and transmission
    coefficients on both faces. Reflection leaves from the incident face
    and transmission leaves from the opposite face. Their fractions must
    sum to at most 1. Each input accepts a scalar or one value per triangle.
    Use metres for the mesh and clipping distances to obtain W and W/m².

    `solar_directions` supplies (ray travel direction, direct normal
    irradiance in W/m²) pairs. PlantGL projects each beam to its first
    visible triangle. Reflected and diffusely transmitted sunlight starts
    the bounce calculation alongside intrinsic emission. `solar_resolution`
    is the pixel size in metres for the default Z-buffer projection.
    `sky_diffuse_irradiance` is diffuse horizontal irradiance in W/m²,
    distributed over PlantGL's standard overcast sky. Sky and sun share
    the projection settings. Their first-hit powers are reported separately
    from indirect light generated by subsequent bounces.
    `sky_threads` limits parallel Z-buffer projections of the sky directions.

    An outgoing packet's form factors are sampled at the triangle centroid.
    `hemicube_threads` limits the number of workers that compute independent
    rows of form factors. The default of 1 computes them sequentially.
    One bounce propagates all power pending at the start of the iteration.
    Stop once the unshot power is within `tolerance` times the total emitted
    and intercepted external power, or after `max_bounces`. Radiosity includes
    power generated on the final bounce even when it has yet to be propagated.
    """
    resolution = index(resolution)
    max_bounces = index(max_bounces)
    hemicube_threads = index(hemicube_threads)
    sky_threads = index(sky_threads)
    if hemicube_threads < 1:
        raise ValueError("hemicube_threads must be at least 1")
    if sky_threads < 1:
        raise ValueError("sky_threads must be at least 1")
    if not 2 <= resolution <= 65534 or resolution % 2:
        raise ValueError("resolution must be an even integer between 2 and 65534")
    if max_bounces < 0 or not isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("max_bounces must be nonnegative and tolerance must be in [0, 1)")
    if not isfinite(near) or near <= 0 or (far is not None and (not isfinite(far) or far <= near)):
        raise ValueError("clipping distances must satisfy 0 < near < far")
    if not isfinite(sky_diffuse_irradiance) or sky_diffuse_irradiance < 0:
        raise ValueError("sky_diffuse_irradiance must be finite and nonnegative")

    if not isinstance(points, Point3Array):
        points = Point3Array(list(points))
    if any(not all(map(isfinite, point)) for point in points):
        raise ValueError("mesh vertices must be finite")
    if not isinstance(triangles, Index3Array):
        triangles = Index3Array(list(triangles))
    count = len(triangles)
    emission = _triangle_values(emission, count, "emission")
    reflectance = _triangle_values(reflectance, count, "reflectance", 1)
    two_sided = transmittance is not None
    transmission = (
        _triangle_values(transmittance, count, "transmittance", 1)
        if two_sided else (0.0,) * count
    )
    if any(rho + tau > 1 for rho, tau in zip(reflectance, transmission)):
        raise ValueError("reflectance plus transmittance must not exceed 1")

    centers, normals, areas = [], [], []
    for triangle in triangles:
        if any(vertex >= len(points) for vertex in triangle):
            raise ValueError("triangle vertex index is outside the mesh")
        a, b, c = (points[vertex] for vertex in triangle)
        normal = cross(b - a, c - a)
        double_area = norm(normal)
        if not isfinite(double_area) or double_area <= 0:
            raise ValueError("radiosity requires nondegenerate triangles")
        areas.append(double_area / 2)
        centers.append((a + b + c) / 3)
        normals.append(normal / double_area)

    # Each triangle has two consecutive face indices, front then back.
    pending = [0.0] * (2 * count)
    pending[::2] = [area * value for area, value in zip(areas, emission)]
    emitted = fsum(pending)
    if not isfinite(emitted):
        raise ValueError("total emitted power must be finite")
    direct = [0.0] * (2 * count)
    sky = direct.copy()
    if solar_directions is not None or sky_diffuse_irradiance:
        scene, projection_options = _external_scene(points, triangles, solar_method, solar_resolution)
        if solar_directions is not None:
            direct = _directional_face_powers(
                scene, normals, solar_directions, solar_method, projection_options,
            )
        if sky_diffuse_irradiance:
            sky_directions = estimate_dir_vectors(soc_sky_sources(sky_diffuse_irradiance))
            if solar_method == eZBufferProjection:
                rays, irradiances = zip(*sky_directions)
                sky = list(algo.directionalFacePowersBatch(
                    points, triangles, Point3Array(normals), Point3Array(rays),
                    RealArray(irradiances), pixelSize=solar_resolution or 0, threads=sky_threads,
                ))
            else:
                sky = _directional_face_powers(
                    scene, normals, sky_directions, solar_method, projection_options,
                )
    direct_absorbed = [0.0] * count
    sky_absorbed = [0.0] * count
    for face, (sun_power, sky_power) in enumerate(zip(direct, sky)):
        power = sun_power + sky_power
        if power == 0:
            continue
        target, back = divmod(face, 2)
        rho = reflectance[target] if two_sided or not back else 0.0
        tau = transmission[target]
        absorptance = 1 - (rho + tau)
        direct_absorbed[target] += sun_power * absorptance
        sky_absorbed[target] += sky_power * absorptance
        pending[face] += power * rho
        pending[face ^ 1] += power * tau

    threshold = tolerance * (emitted + fsum(direct) + fsum(sky))
    transport = _pglalgo._RadiosityTransport(
        points, triangles, Point3Array(centers), Point3Array(normals),
        reflectance, transmission, two_sided, resolution, near, far, hemicube_threads,
    )
    outgoing, indirect_incident, indirect_absorbed, escaped, remaining, bounces = (
        transport.propagate(pending, max_bounces, threshold)
    )

    front_radiosity = tuple(power / area for power, area in zip(outgoing[::2], areas))
    back_radiosity = tuple(power / area for power, area in zip(outgoing[1::2], areas))
    return RadiosityResult(
        radiosity=tuple(front + back for front, back in zip(front_radiosity, back_radiosity)),
        incident_power=tuple(front + back + sky_front + sky_back + indirect
                             for front, back, sky_front, sky_back, indirect in
                             zip(direct[::2], direct[1::2], sky[::2], sky[1::2], indirect_incident)),
        absorbed_power=tuple(sun + sky + indirect for sun, sky, indirect in
                             zip(direct_absorbed, sky_absorbed, indirect_absorbed)),
        emitted_power=emitted,
        escaped_power=escaped,
        remaining_power=remaining,
        bounces=bounces,
        converged=remaining <= threshold,
        front_radiosity=front_radiosity,
        back_radiosity=back_radiosity,
        triangle_area=tuple(areas),
        direct_front_power=tuple(direct[::2]),
        direct_back_power=tuple(direct[1::2]),
        direct_absorbed_power=tuple(direct_absorbed),
        indirect_incident_power=tuple(indirect_incident),
        indirect_absorbed_power=tuple(indirect_absorbed),
        sky_front_power=tuple(sky[::2]),
        sky_back_power=tuple(sky[1::2]),
        sky_absorbed_power=tuple(sky_absorbed),
    )
