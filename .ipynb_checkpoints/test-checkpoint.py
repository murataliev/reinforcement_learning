import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('mujoco_menagerie/universal_robots_ur5e/scene.xml')
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()