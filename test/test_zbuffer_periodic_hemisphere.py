import pytest

import openalea.plantgl.all as pgl


@pytest.mark.parametrize("depth_step", [0, 1.5])
@pytest.mark.parametrize("direction", [-1, 1])
def test_vertical_periodization_matches_translated_geometry(depth_step, direction):
    size = 12
    actual = pgl.ZBufferEngine(size, size, renderingStyle=pgl.eIdBased, multithreaded=False)
    expected = pgl.ZBufferEngine(size, size, renderingStyle=pgl.eIdBased, multithreaded=False)
    for engine in (actual, expected):
        engine.setOrthographicCamera(-6, 6, -6, 6, 1, 10)
        engine.lookAt((0, 0, 0), (0, 0, -1), (0, 1, 0))

    actual.process(pgl.PointSet([(0, 0, -5)]), pgl.Material(), 8)
    actual.periodizeBuffer(0, direction, direction * depth_step)
    for y in range(size):
        depth = 5 + (y - 6) * depth_step
        if 1 <= depth <= 10:
            expected.process(pgl.PointSet([(0, 6 - y, -depth)]), pgl.Material(), 8)

    actual_depth = actual.getDepthBuffer()
    expected_depth = expected.getDepthBuffer()
    for x in range(size):
        for y in range(size):
            assert actual_depth[x, y] == expected_depth[x, y], (x, y)


@pytest.mark.parametrize("points, near, far", [
    ([(-5, 6, -7), (-9, 9, -5), (6, 6, -4)], 0, 100),
    ([(-1, 7, 2), (4, -3, 1), (-3, 0, -1)], 0, 100),
    ([(-3, -3, -2), (3, -3, -2), (0, 4, -2)], 2.2, 3),
])
def test_hemisphere_matches_exhaustive_pixel_rays(points, near, far):
    size = 24
    engine = pgl.ZBufferEngine(size, size, defaultId=0, renderingStyle=pgl.eIdBased, multithreaded=False)
    engine.setHemisphericCamera(near, far)
    engine.lookAt((0, 0, 0), (0, 0, -1), (0, 1, 0))
    camera = engine.camera()
    triangle = [pgl.Vector3(point) for point in points]

    depth = engine.getDepthBuffer()
    background = {(x, y): depth[x, y] for x in range(size) for y in range(size)}
    engine.process(pgl.TriangleSet(triangle, [(0, 1, 2)]), pgl.Material(), 8)

    identifiers = engine.getIdBufferAsImage()
    hits = 0
    for (x, y), expected in background.items():
        expected_id = 0
        sample = pgl.Vector3(x + 0.5, y + 0.5, 1)
        ndc = camera.rasterToNDC(sample, size, size)
        if ndc.x**2 + ndc.y**2 <= 1:
            direction = camera.rasterToCamera(sample, size, size).normed()
            hit = pgl.Ray((0, 0, 0), direction).intersect(*triangle)
            if isinstance(hit, pgl.Vector3) and near <= pgl.norm(hit) <= far:
                expected = pgl.norm(hit)
                expected_id = 8
                hits += 1
        assert depth[x, y] == pytest.approx(expected), (x, y)
        assert identifiers.getPixelAt(x, y).toUint(pgl.eARGB) == expected_id, (x, y)
    assert hits > 0
