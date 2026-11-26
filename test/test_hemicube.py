from math import atan, pi, sqrt

import pytest

import openalea.plantgl.all as pgl


def square_factor(half_width_over_distance):
    ratio = half_width_over_distance
    side = sqrt(1 + ratio * ratio)
    return (4 / pi) * (ratio / side) * atan(ratio / side)


@pytest.mark.parametrize("receiver_half_width, blocker_half_width, rotated", [
    (1, None, False),
    (2, None, False),
    (2, 0.25, False),
    (2, 1, False),
    (2, 0.25, True),
])
def test_hemicube_parallel_plates(receiver_half_width, blocker_half_width, rotated):
    points = []
    triangles = []

    def plate(half_width, z):
        start = len(points)
        points.extend([
            (-half_width, -half_width, z), (half_width, -half_width, z),
            (half_width, half_width, z), (-half_width, half_width, z),
        ])
        triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])

    plate(0.1, 0)
    plate(receiver_half_width, 1)
    if blocker_half_width is not None:
        plate(blocker_half_width, 0.5)

    center, normal = (0, 0, 0), (0, 0, 1)
    if rotated:
        center, normal = (10, 20, 30), (1, 0, 0)
        points = [(z + 10, x + 20, y + 30) for x, y, z in points]

    factors, escaped = pgl.hemicubeFormFactors(
        pgl.Point3Array(points), pgl.Index3Array(triangles),
        center, normal, resolution=128, emitterId=0,
    )
    receiver = factors.get(2, 0) + factors.get(3, 0)
    blocker = factors.get(4, 0) + factors.get(5, 0)
    expected_receiver = square_factor(receiver_half_width)
    expected_blocker = square_factor(2 * blocker_half_width) if blocker_half_width is not None else 0

    assert 0 not in factors and 1 not in factors
    assert all(value > 0 for value in factors.values())
    assert sum(factors.values()) + escaped == pytest.approx(1, abs=2e-5)
    assert receiver == pytest.approx(expected_receiver - expected_blocker, abs=0.008)
    assert blocker == pytest.approx(expected_blocker, abs=0.008)
    assert escaped == pytest.approx(1 - expected_receiver, abs=0.008)
    if blocker_half_width == 1:
        assert receiver == 0


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_hemicube_batch_matches_single_rows_and_clears_visibility(threads):
    points = pgl.Point3Array([
        (-1, -1, 1), (1, -1, 1), (0, 1, 1),
        (-2, -2, 2), (2, -2, 2), (0, 2, 2),
    ])
    triangles = pgl.Index3Array([(0, 1, 2), (3, 4, 5)])
    centers = pgl.Point3Array([(0, 0, 0)] * 4)
    normals = pgl.Point3Array([(0, 0, 1), (0, 0, -1), (0, 0, 1), (0, 0, 1)])
    emitter_ids = pgl.Index([1, 0, 0, 1])
    rows = pgl.hemicubeFormFactorsBatch(
        points, triangles, centers, normals, resolution=32, emitterIds=emitter_ids, threads=threads,
    )
    expected = [
        pgl.hemicubeFormFactors(points, triangles, center, normal, resolution=32, emitterId=emitter)
        for center, normal, emitter in zip(centers, normals, emitter_ids)
    ]
    assert rows == expected
    assert set(rows[0][0]) == {0}
    assert rows[1] == ({}, 1)
    assert set(rows[2][0]) == {1}
    assert rows[3] == rows[0]
    assert all(sum(factors.values()) + escape == pytest.approx(1) for factors, escape in rows)
