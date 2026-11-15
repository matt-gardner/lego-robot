import mujoco

from so101.settled_piles import support_graph


def test_contact_graph_distinguishes_supported_stack_from_floating_body():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .1"/>
      <body name="bottom" pos="0 0 .0099">
        <freejoint/><geom type="box" size=".01 .01 .01"/>
      </body>
      <body name="top" pos="0 0 .0298">
        <freejoint/><geom type="box" size=".01 .01 .01"/>
      </body>
      <body name="floating" pos=".1 0 .1">
        <freejoint/><geom type="box" size=".01 .01 .01"/>
      </body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = support_graph(model, data, ["bottom", "top", "floating"])
    assert result["table_connected_objects"] == ["bottom", "top"]
    assert result["unsupported_objects"] == ["floating"]
    assert result["object_object_edges"] == 1


def test_static_obstacles_count_as_ground():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .1"/>
      <geom name="pile_obstacle_0" type="box" pos=".1 0 .01" size=".02 .02 .01"/>
      <body name="on_step" pos=".1 0 .0299">
        <freejoint/><geom type="box" size=".01 .01 .01"/>
      </body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert support_graph(model, data, ["on_step"])["table_connected_objects"] == ["on_step"]


def test_micrometre_resting_gaps_count_as_support_without_contacts():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .1"/>
      <body name="resting" pos="0 0 .010005">
        <freejoint/><geom type="box" size=".01 .01 .01"/>
      </body>
      <body name="stacked" pos="0 0 .03001">
        <freejoint/><geom type="box" size=".01 .01 .01"/>
      </body>
      <body name="hovering" pos=".1 0 .012">
        <freejoint/><geom type="box" size=".01 .01 .01"/>
      </body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    result = support_graph(model, data, ["resting", "stacked", "hovering"])
    assert result["table_connected_objects"] == ["resting", "stacked"]
    assert result["unsupported_objects"] == ["hovering"]
