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
