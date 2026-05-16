import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root
sys.path.append(PROJECT_ROOT)

def make_robot(params, assets_dir, env_name="ant"):
    """Generate robot XML for different environments based on parameters."""
    env_name_lower = env_name.lower()

    if env_name_lower == "ant":
        return _make_ant_robot(params, assets_dir)
    elif env_name_lower == "half_cheetah":
        return _make_half_cheetah_robot(params, assets_dir)
    elif env_name_lower == "hopper":
        return _make_hopper_robot(params, assets_dir)
    elif env_name_lower == "swimmer":
        return _make_swimmer_robot(params, assets_dir)
    elif env_name_lower == "walker2d":
        return _make_walker2d_robot(params, assets_dir)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

def _make_ant_robot(params, assets_dir):
    """Generate ant robot XML."""
    output_path = "ant_modified.xml"
    full_path = os.path.join(assets_dir, output_path)

    xml_output = f"""
<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.02"/>
  <custom>
    <numeric data="0.0 0.0 {params[0]} 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01"
             rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true"
           exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor"
          pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 {params[0]}">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="{params[0]}" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>

      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {params[1]} {params[2]} 0.0" name="aux_1_geom" size="{params[7]}" type="capsule"/>
        <body name="aux_1" pos="{params[1]} {params[2]} 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {params[3]} {params[4]} 0.0" name="left_leg_geom" size="{params[8]}" type="capsule" />
          <body pos="{params[3]} {params[4]} 0" >
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {params[5]} {params[6]} 0.0" name="left_ankle_geom" size="{params[9]}" type="capsule"/>
          </body>
        </body>
      </body>

      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{params[1]} {params[2]} 0.0" name="aux_2_geom" size="{params[7]}" type="capsule"/>
        <body name="aux_2" pos="-{params[1]} {params[2]} 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{params[3]} {params[4]} 0.0" name="right_leg_geom" size="{params[8]}" type="capsule" />
          <body pos="-{params[3]} {params[4]} 0" >
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{params[5]} {params[6]} 0.0" name="right_ankle_geom" size="{params[9]}" type="capsule"/>
          </body>
        </body>
      </body>

      <body name="left_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{params[1]} -{params[2]} 0.0" name="aux_3_geom" size="{params[7]}" type="capsule"/>
        <body name="aux_3" pos="-{params[1]} -{params[2]} 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{params[3]} -{params[4]} 0.0" name="back_leg_geom" size="{params[8]}" type="capsule" />
          <body pos="-{params[3]} -{params[4]} 0" >
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{params[5]} -{params[6]} 0.0" name="third_ankle_geom" size="{params[9]}" type="capsule"/>
          </body>
        </body>
      </body>

      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {params[1]} -{params[2]} 0.0" name="aux_4_geom" size="{params[7]}" type="capsule"/>
        <body name="aux_4" pos="{params[1]} -{params[2]} 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {params[3]} -{params[4]} 0.0" name="rightback_leg_geom" size="{params[8]}" type="capsule" />
          <body pos="{params[3]} -{params[4]} 0" >
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {params[5]} -{params[6]} 0.0" name="fourth_ankle_geom" size="{params[9]}" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
  </actuator>
</mujoco>
"""

    os.makedirs(assets_dir, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(xml_output.strip())

def _make_half_cheetah_robot(params, assets_dir):
    """Generate half cheetah robot XML."""
    output_path = "half_cheetah_modified.xml"
    full_path = os.path.join(assets_dir, output_path)

    # Auto-raise the torso if any leg endpoint would start below the ground.
    # Keep the original morphology and joint ranges; only translate the body up.
    # Compute cumulative z offsets down each kinematic chain to detect ground penetration.
    back_thigh_z = params[5]
    back_shin_z = params[5] + params[7]
    back_foot_z = params[5] + params[7] + params[9]
    front_thigh_z = params[11]
    front_shin_z = params[11] + params[13]
    front_foot_z = params[11] + params[13] + params[15]
    min_z = min(
        0.0,  # torso origin
        back_thigh_z,
        back_shin_z,
        back_foot_z,
        front_thigh_z,
        front_shin_z,
        front_foot_z,
    )
    base_torso_z = 0.7
    clearance = 0.05
    torso_z = base_torso_z if min_z >= -clearance else (-min_z + clearance)

    xml_output = f"""
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 {torso_z}">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="ignorex" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="ignorez" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="ignorey" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="{params[0]} 0 0 {params[1]} 0 0" name="torso" size="{params[16]}" type="capsule"/>
      <geom fromto="{params[1]} 0 0 {params[2]} 0 {params[3]}" name="head" size="{params[17]}" type="capsule"/>

      <body name="bthigh" pos="{params[0]} 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom fromto="0 0 0 {params[4]} 0 {params[5]}" name="bthigh" size="{params[18]}" type="capsule"/>
        <body name="bshin" pos="{params[4]} 0 {params[5]}">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom fromto="0 0 0 {params[6]} 0 {params[7]}" name="bshin" rgba="0.9 0.6 0.6 1" size="{params[19]}" type="capsule"/>
          <body name="bfoot" pos="{params[6]} 0 {params[7]}">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom fromto="0 0 0 {params[8]} 0 {params[9]}" name="bfoot" rgba="0.9 0.6 0.6 1" size="{params[20]}" type="capsule"/>
          </body>
        </body>
      </body>

      <body name="fthigh" pos="{params[1]} 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1.5 0.8" stiffness="180" type="hinge"/>
        <geom fromto="0 0 0 {params[10]} 0 {params[11]}" name="fthigh" size="{params[21]}" type="capsule"/>
        <body name="fshin" pos="{params[10]} 0 {params[11]}">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 1.1" stiffness="120" type="hinge"/>
          <geom fromto="0 0 0 {params[12]} 0 {params[13]}" name="fshin" rgba="0.9 0.6 0.6 1" size="{params[22]}" type="capsule"/>
          <body name="ffoot" pos="{params[12]} 0 {params[13]}">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-3.1 -0.3" stiffness="60" type="hinge"/>
            <geom fromto="0 0 0 {params[14]} 0 {params[15]}" name="ffoot" rgba="0.9 0.6 0.6 1" size="{params[23]}" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>
"""

    os.makedirs(assets_dir, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(xml_output.strip())

def _make_hopper_robot(params, assets_dir):
    """Generate hopper robot XML from a vertical capsule-chain template.

    Parameters are interpreted as absolute world z-coordinates (converted to relative):
    - param1: Upper attachment point of torso (highest, positive)
    - param2: Lower attachment point of torso and upper attachment point of thigh (less than param1, positive)
    - param3: Lower attachment point of thigh and upper attachment point of leg (less than param2, positive)
    - param4: Lower attachment point of leg and upper attachment point of foot (less than param3, positive)
    - param5: Left end of foot (x-coordinate, positive)
    - param6: Right end of foot (x-coordinate, negative)
    - param7: Torso capsule radius
    - param8: Thigh capsule radius
    - param9: Leg capsule radius
    - param10: Foot capsule radius
    """
    output_path = "hopper_modified.xml"
    full_path = os.path.join(assets_dir, output_path)

    # Convert absolute world coordinates to relative body coordinates
    # Torso body positioned at param1 (top)
    torso_body_z = params[0]

    # Torso geom extends from body origin (param1) down to param2
    torso_bottom_rel = params[1] - params[0]  # Negative value

    # Thigh body positioned at param2 (relative to torso)
    thigh_body_z_rel = params[1] - params[0]  # Negative value

    # Thigh geom extends from thigh body origin down to param3
    thigh_bottom_rel = params[2] - params[1]  # Negative value

    # Leg body positioned at param3 (relative to thigh)
    leg_body_z_rel = params[2] - params[1]  # Negative value

    # Leg geom extends from leg body origin down to param4
    leg_bottom_rel = params[3] - params[2]  # Negative value

    # Foot body positioned at param4 (relative to leg)
    foot_body_z_rel = params[3] - params[2]  # Negative value

    xml_output = f"""
<mujoco model="hopper">
  <compiler angle="degree" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1" solimp=".8 .8 .01" solref=".02 1"/>
    <motor ctrllimited="true" ctrlrange="-.4 .4"/>
  </default>
  <option integrator="RK4" timestep="0.002"/>
  <visual>
    <map znear="0.02"/>
  </visual>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
    <body name="torso" pos="0 0 {torso_body_z}">
      <camera name="track" mode="trackcom" pos="0 -3 -0.25" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="ignore1" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="ignore2" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="ignore3" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="0 0 0 0 0 {torso_bottom_rel}" name="torso_geom" size="{params[6]}" type="capsule" friction="0.9"/>
      <body name="thigh" pos="0 0 {thigh_body_z_rel}">
        <joint axis="0 -1 0" name="thigh_joint" pos="0 0 0" range="-150 0" type="hinge"/>
        <geom fromto="0 0 0 0 0 {thigh_bottom_rel}" name="thigh_geom" size="{params[7]}" type="capsule" friction="0.9"/>
        <body name="leg" pos="0 0 {leg_body_z_rel}">
          <joint axis="0 -1 0" name="leg_joint" pos="0 0 0" range="-150 0" type="hinge"/>
          <geom fromto="0 0 0 0 0 {leg_bottom_rel}" name="leg_geom" size="{params[8]}" type="capsule" friction="0.9"/>
          <body name="foot" pos="0 0 {foot_body_z_rel}">
            <joint axis="0 -1 0" name="foot_joint" pos="0 0 0" range="-45 45" type="hinge"/>
            <geom fromto="{params[4]} 0 0 {params[5]} 0 0" name="foot_geom" size="{params[9]}" type="capsule" friction="2.0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="thigh_joint"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="leg_joint"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="foot_joint"/>
  </actuator>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0"
            width="100" height="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
</mujoco>
"""

    os.makedirs(assets_dir, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(xml_output.strip())

def _make_swimmer_robot(params, assets_dir):
    """Generate swimmer robot XML."""
    output_path = "swimmer_modified.xml"
    full_path = os.path.join(assets_dir, output_path)

    xml_output = f"""
<mujoco model="swimmer">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option density="4000" integrator="RK4" timestep="0.01" viscosity="0.1"/>
  <default>
    <geom conaffinity="0" condim="1" contype="0" material="geom" rgba="0.8 0.6 .4 1"/>
    <joint armature='0.1'  />
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="30 30" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom condim="3" material="MatPlane" name="floor" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1" size="40 40 0.1" type="plane"/>
    <body name="torso" pos="0 0 0">
      <camera name="track" mode="trackcom" pos="0 -3 3" xyaxes="1 0 0 0 1 1"/>
      <geom density="1000" fromto="{params[0]} 0 0 0 0 0" size="{params[3]}" type="capsule"/>
      <joint axis="1 0 0" name="slider1" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" name="slider2" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" name="free_body_rot" pos="0 0 0" type="hinge"/>
      <body name="mid" pos="0 0 0">
        <geom density="1000" fromto="0 0 0 -{params[1]} 0 0" size="{params[4]}" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="motor1_rot" pos="0 0 0" range="-100 100" type="hinge"/>
        <body name="back" pos="-{params[1]} 0 0">
          <geom density="1000" fromto="0 0 0 -{params[2]} 0 0" size="{params[5]}" type="capsule"/>
          <joint axis="0 0 1" limited="true" name="motor2_rot" pos="0 0 0" range="-100 100" type="hinge"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="motor1_rot"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="motor2_rot"/>
  </actuator>
</mujoco>
"""

    os.makedirs(assets_dir, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(xml_output.strip())

def _make_walker2d_robot(params, assets_dir):
    """Generate walker2d robot XML from a vertical capsule-chain template.

    Parameters are interpreted as absolute world z-coordinates (converted to relative):
    - param1: Upper attachment point of torso (highest, positive)
    - param2: Lower attachment point of torso and upper attachment point of thigh (less than param1, positive)
    - param3: Lower attachment point of thigh and upper attachment point of leg (less than param2, positive)
    - param4: Lower attachment point of leg and upper attachment point of foot (less than param3, positive)
    - param5: Left end of foot (x-coordinate, positive)
    - param6: Right end of foot (x-coordinate, negative)
    - param7: Torso capsule radius
    - param8: Thigh capsule radius
    - param9: Leg capsule radius
    - param10: Foot capsule radius
    """
    output_path = "walker2d_modified.xml"
    full_path = os.path.join(assets_dir, output_path)

    # Convert absolute world coordinates to relative body coordinates
    # Torso body positioned at param1 (top)
    torso_body_z = params[0]

    # Torso geom extends from body origin (param1) down to param2
    torso_bottom_rel = params[1] - params[0]  # Negative value

    # Thigh body positioned at param2 (relative to torso)
    thigh_body_z_rel = params[1] - params[0]  # Negative value

    # Thigh geom extends from thigh body origin down to param3
    thigh_bottom_rel = params[2] - params[1]  # Negative value

    # Leg body positioned at param3 (relative to thigh)
    leg_body_z_rel = params[2] - params[1]  # Negative value

    # Leg geom extends from leg body origin down to param4
    leg_bottom_rel = params[3] - params[2]  # Negative value

    # Foot body positioned at param4 (relative to leg)
    foot_body_z_rel = params[3] - params[2]  # Negative value

    xml_output = f"""
<mujoco model="walker2d">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
      <joint armature="0.01" damping=".1" limited="true"/>
      <geom conaffinity="0" condim="3" contype="1" density="1000" friction=".7 .1 .1" rgba="0.8 0.6 .4 1"/>
    </default>
    <option integrator="RK4" timestep="0.002"/>
    <worldbody>
      <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
      <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
      <body name="torso" pos="0 0 {torso_body_z}">
        <camera name="track" mode="trackcom" pos="0 -3 -0.25" xyaxes="1 0 0 0 0 1"/>
        <joint armature="0" axis="1 0 0" damping="0" limited="false" name="ignore1" pos="0 0 0" stiffness="0" type="slide"/>
        <joint armature="0" axis="0 0 1" damping="0" limited="false" name="ignore2" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
        <joint armature="0" axis="0 1 0" damping="0" limited="false" name="ignore3" pos="0 0 0" stiffness="0" type="hinge"/>
        <geom fromto="0 0 0 0 0 {torso_bottom_rel}" name="torso_geom" size="{params[6]}" type="capsule" friction="0.9"/>
        <body name="thigh" pos="0 0 {thigh_body_z_rel}">
          <joint axis="0 -1 0" name="thigh_joint" pos="0 0 0" range="-150 0" type="hinge"/>
          <geom fromto="0 0 0 0 0 {thigh_bottom_rel}" name="thigh_geom" size="{params[7]}" type="capsule" friction="0.9"/>
          <body name="leg" pos="0 0 {leg_body_z_rel}">
            <joint axis="0 -1 0" name="leg_joint" pos="0 0 0" range="-150 0" type="hinge"/>
            <geom fromto="0 0 0 0 0 {leg_bottom_rel}" name="leg_geom" size="{params[8]}" type="capsule" friction="0.9"/>
            <body name="foot" pos="0 0 {foot_body_z_rel}">
              <joint axis="0 -1 0" name="foot_joint" pos="0 0 0" range="-45 45" type="hinge"/>
              <geom fromto="{params[4]} 0 0 {params[5]} 0 0" name="foot_geom" size="{params[9]}" type="capsule" friction="1.9"/>
            </body>
          </body>
        </body>
        <body name="thigh_left" pos="0 0 {thigh_body_z_rel}">
          <joint axis="0 -1 0" name="thigh_left_joint" pos="0 0 0" range="-150 0" type="hinge"/>
          <geom fromto="0 0 0 0 0 {thigh_bottom_rel}" name="thigh_left_geom" size="{params[7]}" type="capsule" friction="0.9"/>
          <body name="leg_left" pos="0 0 {leg_body_z_rel}">
            <joint axis="0 -1 0" name="leg_left_joint" pos="0 0 0" range="-150 0" type="hinge"/>
            <geom fromto="0 0 0 0 0 {leg_bottom_rel}" name="leg_left_geom" size="{params[8]}" type="capsule" friction="0.9"/>
            <body name="foot_left" pos="0 0 {foot_body_z_rel}">
              <joint axis="0 -1 0" name="foot_left_joint" pos="0 0 0" range="-45 45" type="hinge"/>
              <geom fromto="{params[4]} 0 0 {params[5]} 0 0" name="foot_left_geom" size="{params[9]}" type="capsule" friction="1.9"/>
            </body>
          </body>
        </body>
      </body>
    </worldbody>

    <actuator>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="thigh_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="leg_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="foot_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="thigh_left_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="leg_left_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="foot_left_joint"/>
    </actuator>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0"
            width="100" height="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
      </asset>
  </mujoco>
"""

    os.makedirs(assets_dir, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(xml_output.strip())

if __name__ == "__main__":
    # Test with sample parameters
    params = [0.25, 0.2, 0.2, 0.2, 0.3, 0.1, 0.1, 0.02, 0.01, 0.03]
    make_robot(params, "assets", "hopper")
