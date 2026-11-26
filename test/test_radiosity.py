from math import fsum

import pytest

import openalea.plantgl.all as pgl
from openalea.plantgl.light import solve_radiosity


@pytest.fixture
def facing_triangles():
    points = pgl.Point3Array([
        (-2, -2, 0), (2, -2, 0), (0, 2, 0),
        (-2, -2, 1), (0, 2, 1), (2, -2, 1),
    ])
    triangles = pgl.Index3Array([(0, 1, 2), (3, 4, 5)])
    return points, triangles


def check_energy(result):
    assert result.emitted_power + result.solar_intercepted_power + result.sky_intercepted_power == pytest.approx(
        fsum(result.absorbed_power) + result.escaped_power + result.remaining_power,
        rel=1e-10,
    )


def test_radiosity_matches_closed_form_for_two_facing_triangles(facing_triangles):
    points, triangles = facing_triangles
    reflectance = (0.65, 0.8)
    resolution = 64
    row0, escape0 = pgl.hemicubeFormFactors(
        points, triangles, (0, -2 / 3, 0), (0, 0, 1), resolution=resolution, emitterId=0,
    )
    row1, escape1 = pgl.hemicubeFormFactors(
        points, triangles, (0, -2 / 3, 1), (0, 0, -1), resolution=resolution, emitterId=1,
    )
    assert set(row0) == {1} and set(row1) == {0}
    f01, f10 = row0[1], row1[0]

    result = solve_radiosity(
        points, triangles, (1, 0), reflectance, resolution=resolution, tolerance=1e-10,
    )
    assert result.bounces > 3
    assert result.converged
    assert result.remaining_power <= 1e-10 * result.emitted_power
    assert result.emitted_power == 8

    # Each round trip multiplies the emitted power by this factor.
    round_trip = reflectance[0] * f10 * reflectance[1] * f01
    power0 = 8 / (1 - round_trip)
    power1 = power0 * f01 * reflectance[1]
    incident = (power1 * f10, power0 * f01)

    assert result.radiosity == pytest.approx((power0 / 8, power1 / 8), rel=1e-9)
    assert result.incident_power == pytest.approx(incident, rel=1e-9)
    assert result.absorbed_power == pytest.approx(
        tuple(power * (1 - rho) for power, rho in zip(incident, reflectance)), rel=1e-9,
    )
    assert result.escaped_power == pytest.approx(escape0 * power0 + escape1 * power1, rel=1e-9)
    check_energy(result)

    stopped = solve_radiosity(
        points, triangles, (1, 0), reflectance, resolution=resolution, max_bounces=3,
    )
    assert stopped.bounces == 3
    assert not stopped.converged
    assert stopped.remaining_power == pytest.approx(8 * round_trip * f01 * reflectance[1])
    assert stopped.incident_power == pytest.approx(
        (8 * f01 * reflectance[1] * f10, 8 * (1 + round_trip) * f01),
    )
    check_energy(stopped)


def test_radiosity_absorbs_hits_on_the_back_of_an_opaque_triangle(facing_triangles):
    points, _ = facing_triangles
    triangles = [(0, 1, 2), (3, 5, 4)]
    result = solve_radiosity(points, triangles, (1, 0), 0.8, resolution=64)

    assert result.bounces == 1
    assert result.converged
    assert result.incident_power[1] > 0
    assert result.absorbed_power == result.incident_power
    assert result.radiosity == (1, 0)
    check_energy(result)


def test_radiosity_reflects_and_transmits_from_both_leaf_faces():
    points = pgl.Point3Array([
        (x, y, z)
        for z in range(3) for x, y in [(-2, -2), (2, -2), (0, 2)]
    ])
    triangles = pgl.Index3Array([(0, 1, 2), (3, 5, 4), (6, 8, 7)])
    resolution = 64
    rho = (0.5, 0.2, 0.6)
    tau = (0, 0.65, 0)

    def factor(source, target, direction):
        row, _ = pgl.hemicubeFormFactors(
            points, triangles, (0, -2 / 3, source), (0, 0, direction),
            resolution=resolution, emitterId=source,
        )
        assert set(row) == {target}
        return row[target]

    f01, f10 = factor(0, 1, 1), factor(1, 0, -1)
    f12, f21 = factor(1, 2, 1), factor(2, 1, -1)

    result = solve_radiosity(
        points, triangles, (1, 0, 0), rho, transmittance=tau,
        resolution=resolution, max_bounces=3,
    )
    # The middle leaf receives on its front, then sends power back to the
    # emitter and through to the far triangle. Both return power on bounce 3.
    first = 8 * f01
    second_bottom = first * rho[1] * f10
    second_top = first * tau[1] * f12
    third_front = second_bottom * rho[0] * f01
    third_back = second_top * rho[2] * f21
    assert result.bounces == 3
    assert not result.converged
    assert result.incident_power == pytest.approx(
        (second_bottom, first + third_front + third_back, second_top),
    )
    assert result.front_radiosity == pytest.approx((
        (8 + second_bottom * rho[0]) / 8,
        ((first + third_front) * rho[1] + third_back * tau[1]) / 8,
        second_top * rho[2] / 8,
    ))
    assert result.back_radiosity == pytest.approx((
        0, ((first + third_front) * tau[1] + third_back * rho[1]) / 8, 0,
    ))
    assert result.remaining_power == pytest.approx((third_front + third_back) * (rho[1] + tau[1]))
    check_energy(result)

    converged = solve_radiosity(
        points, triangles, (1, 0, 0), rho, transmittance=tau, resolution=resolution,
    )
    assert converged.converged and converged.bounces > 3
    assert converged.remaining_power < result.remaining_power
    check_energy(converged)


def test_radiosity_zero_transmittance_enables_back_reflection(facing_triangles):
    points, _ = facing_triangles
    result = solve_radiosity(
        points, [(0, 1, 2), (3, 5, 4)], (1, 0), 0.5,
        transmittance=0, resolution=64,
    )
    assert result.converged and result.bounces > 1
    assert result.front_radiosity[1] == 0
    assert result.back_radiosity[1] > 0
    check_energy(result)


def test_radiosity_rejects_nonconserving_leaf_coefficients(facing_triangles):
    points, triangles = facing_triangles
    with pytest.raises(ValueError, match="reflectance plus transmittance"):
        solve_radiosity(points, triangles, 1, 0.8, transmittance=(0.1, 0.3))


@pytest.fixture
def small_canopy():
    points = pgl.Point3Array([
        (x + offset, y, z) for offset, z in [(0, 0), (0, 1), (5, 0.4)]
        for x, y in [(-2, -2), (2, -2), (0, 2)]
    ])
    # The upper leaf faces down. Light from above reaches its back.
    return points, pgl.Index3Array([(0, 1, 2), (3, 5, 4), (6, 7, 8)])


@pytest.mark.parametrize("method, solar_options, error", [
    (pgl.eZBufferProjection, {"solar_resolution": 0.025}, 1.5),
    (pgl.eTriangleProjection, {}, 1e-10),
])
def test_solar_radiosity_on_three_leaves(small_canopy, method, solar_options, error):
    if method == pgl.eTriangleProjection and not hasattr(pgl, "DepthSortEngine"):
        pytest.skip("Triangle projection requires CGAL")
    points, triangles = small_canopy
    options = dict(
        emission=0, reflectance=(0.5, 0.2, 0.6), transmittance=(0, 0.65, 0),
        solar_directions=[((0, 0, -1), 10)], solar_method=method, resolution=64,
        **solar_options,
    )

    first = solve_radiosity(points, triangles, max_bounces=0, **options)
    assert first.direct_front_power == pytest.approx((0, 0, 80), abs=error)
    assert first.direct_back_power == pytest.approx((0, 80, 0), abs=error)
    assert first.direct_front_power[0] == first.direct_back_power[0] == 0
    assert first.direct_back_irradiance[1] == pytest.approx(10, abs=error / 8)
    assert first.direct_front_irradiance[2] == pytest.approx(10, abs=error / 8)
    assert first.direct_absorbed_power == pytest.approx((0, 12, 32), abs=error)
    assert first.indirect_incident_power == first.indirect_absorbed_power == (0, 0, 0)
    assert first.remaining_power == pytest.approx(116, abs=2 * error)
    check_energy(first)

    one_bounce = solve_radiosity(points, triangles, max_bounces=1, **options)
    row, _ = pgl.hemicubeFormFactors(
        points, triangles, (0, -2 / 3, 1), (0, 0, -1), resolution=64, emitterId=1,
    )
    expected_lower = first.direct_back_power[1] * 0.65 * row[0]
    assert expected_lower > 0
    assert one_bounce.indirect_incident_power[0] == pytest.approx(expected_lower)
    assert one_bounce.indirect_absorbed_power[0] == pytest.approx(0.5 * expected_lower)
    check_energy(one_bounce)

    result = solve_radiosity(points, triangles, **options)
    assert result.converged and result.bounces > 1
    assert result.emitted_power == 0
    assert result.direct_front_power == first.direct_front_power
    assert result.direct_back_power == first.direct_back_power
    assert result.indirect_incident_power[0] >= expected_lower
    assert result.incident_power[0] == result.indirect_incident_power[0]
    assert result.absorbed_power == pytest.approx(tuple(
        direct + diffuse for direct, diffuse in zip(result.direct_absorbed_power, result.indirect_absorbed_power)
    ))
    check_energy(result)


def test_solar_radiosity_accumulates_beams_on_the_correct_faces(facing_triangles):
    points, _ = facing_triangles
    result = solve_radiosity(
        points[:3], [(0, 1, 2)], 0, 0.6, transmittance=0.3, max_bounces=0,
        solar_directions=[((0, 0, -2), 10), ((0, 0, 3), 5), ((1, 0, 0), 100)],
        solar_resolution=0.025,
    )
    assert result.direct_front_irradiance == pytest.approx((10,), abs=0.2)
    assert result.direct_back_irradiance == pytest.approx((5,), abs=0.1)
    front, back = result.direct_front_power[0], result.direct_back_power[0]
    assert result.front_radiosity[0] == pytest.approx((0.6 * front + 0.3 * back) / 8)
    assert result.back_radiosity[0] == pytest.approx((0.6 * back + 0.3 * front) / 8)
    check_energy(result)


@pytest.mark.parametrize("method, projection_options", [
    (pgl.eZBufferProjection, {"solar_resolution": 0.04}),
    (pgl.eTriangleProjection, {}),
])
def test_sun_sky_and_bounces_on_three_leaves(small_canopy, method, projection_options):
    if method == pgl.eTriangleProjection and not hasattr(pgl, "DepthSortEngine"):
        pytest.skip("Triangle projection requires CGAL")
    points, triangles = small_canopy
    options = dict(
        emission=0, reflectance=(0.5, 0.2, 0.6), transmittance=(0, 0.65, 0),
        solar_method=method, resolution=64, **projection_options,
    )
    sun = dict(solar_directions=[((0, 0, -1), 10)])
    sky = dict(sky_diffuse_irradiance=4)
    before = solve_radiosity(points, triangles, max_bounces=0, **options, **sun, **sky)

    assert before.direct_front_power[0] == before.direct_back_power[0] == 0
    assert before.sky_front_power[0] > 0
    assert before.sky_back_irradiance[1] > 0
    assert before.sky_front_irradiance[2] > 0
    assert before.sky_front_power[1] == before.sky_back_power[2] == 0
    assert before.indirect_incident_power == (0, 0, 0)
    assert before.indirect_absorbed_power == (0, 0, 0)
    assert before.absorbed_power == pytest.approx(tuple(
        solar + diffuse for solar, diffuse in zip(before.direct_absorbed_power, before.sky_absorbed_power)
    ))
    check_energy(before)

    # With a fixed number of bounces the transport is linear in the sources.
    result = solve_radiosity(points, triangles, max_bounces=3, tolerance=0, **options, **sun, **sky)
    sun_only = solve_radiosity(points, triangles, max_bounces=3, tolerance=0, **options, **sun)
    sky_only = solve_radiosity(points, triangles, max_bounces=3, tolerance=0, **options, **sky)
    assert result.direct_front_power == sun_only.direct_front_power
    assert result.direct_back_power == sun_only.direct_back_power
    assert result.sky_front_power == sky_only.sky_front_power
    assert result.sky_back_power == sky_only.sky_back_power
    for field in ("incident_power", "absorbed_power", "indirect_incident_power", "indirect_absorbed_power"):
        expected = tuple(a + b for a, b in zip(getattr(sun_only, field), getattr(sky_only, field)))
        assert getattr(result, field) == pytest.approx(expected)
    assert result.indirect_incident_power[0] > sun_only.indirect_incident_power[0] > 0
    assert result.escaped_power == pytest.approx(sun_only.escaped_power + sky_only.escaped_power)
    assert result.remaining_power == pytest.approx(sun_only.remaining_power + sky_only.remaining_power)
    check_energy(result)

    converged = solve_radiosity(points, triangles, **options, **sun, **sky)
    assert converged.converged and converged.bounces > result.bounces
    check_energy(converged)


@pytest.mark.skipif(not hasattr(pgl, "DepthSortEngine"), reason="Requires CGAL")
def test_sky_diffuse_irradiance_is_horizontal(facing_triangles):
    points, _ = facing_triangles
    result = solve_radiosity(
        points[:3], [(0, 1, 2)], 0, 0, sky_diffuse_irradiance=4,
        solar_method=pgl.eTriangleProjection,
    )
    assert result.sky_front_irradiance == pytest.approx((4,))
    assert result.sky_back_power == (0,)
    assert result.sky_absorbed_power == pytest.approx((32,))
    assert result.solar_intercepted_power == 0 and result.bounces == 0
    check_energy(result)


def test_parallel_hemicube_preserves_radiosity_and_energy(small_canopy):
    points, triangles = small_canopy
    options = dict(
        emission=0, reflectance=(0.5, 0.2, 0.6), transmittance=(0, 0.65, 0),
        solar_directions=[((0, 0, -1), 10)], sky_diffuse_irradiance=4,
        solar_resolution=0.04, resolution=32,
    )
    sequential = solve_radiosity(points, triangles, **options)
    parallel = solve_radiosity(points, triangles, hemicube_threads=4, **options)
    assert parallel == sequential
    check_energy(parallel)


def test_parallel_sky_preserves_radiosity_and_energy(small_canopy):
    points, triangles = small_canopy
    options = dict(
        emission=0, reflectance=(0.5, 0.2, 0.6), transmittance=(0, 0.65, 0),
        solar_directions=[((0, 0, -1), 10)], sky_diffuse_irradiance=4,
        solar_resolution=0.04, resolution=32,
    )
    sequential = solve_radiosity(points, triangles, **options)
    parallel = solve_radiosity(points, triangles, sky_threads=8, **options)
    assert parallel == sequential
    check_energy(parallel)
