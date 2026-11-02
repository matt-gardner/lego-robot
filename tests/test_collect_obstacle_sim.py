from so101.collect_obstacle_sim import edge_indicator_point


def test_edge_indicator_follows_offscreen_box_direction():
    assert edge_indicator_point((2., .5), 320, 240) == (312, 120)
    assert edge_indicator_point((-1., .5), 320, 240) == (8, 120)
    assert edge_indicator_point((.5, 2.), 320, 240) == (160, 232)
    assert edge_indicator_point((.5, -1.), 320, 240) == (160, 8)


def test_edge_indicator_intersects_corner_ray_inside_margin():
    x, y = edge_indicator_point((2., 2.), 320, 240)
    assert 8 <= x <= 312
    assert 8 <= y <= 232
    assert x == 309
    assert y == 232
