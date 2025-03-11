from pydrake.all import (
    RigidTransform,
    MultibodyPlant,
    SceneGraph,
    Cylinder,
    Box,
    UnitInertia,
    SpatialInertia,
    RotationMatrix,
    GeometryInstance,
    MakePhongIllustrationProperties,
    Shape,
    SourceId,
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    AddDefaultVisualization
)
from oculus_drake.teleop_utils import MakeFakeStation
from manipulation.station import load_scenario
from manipulation.scenarios import AddMultibodyTriad
from oculus_drake import FAKE_CALIB_SCENARIO_FILEPATH
import argparse
import json
import numpy as np


def AddAnchoredTriad(
    source_id: SourceId,
    scene_graph: SceneGraph,
    length=0.25,
    radius=0.01,
    opacity=1.0,
    X_FT=RigidTransform(),
    name="frame",
):
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " x-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([1, 0, 0, opacity])
    )
    scene_graph.RegisterAnchoredGeometry(source_id, geom)

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " y-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 1, 0, opacity])
    )
    scene_graph.RegisterAnchoredGeometry(source_id, geom)

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " z-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 0, 1, opacity])
    )
    scene_graph.RegisterAnchoredGeometry(source_id, geom)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--extrinsics_file", type=str, default="config/camera_extrinsics.json")
    args = argparser.parse_args()
    with open(args.extrinsics_file, 'r') as f:
        extrinsics = json.load(f)
    for key in extrinsics.keys():
        extrinsics[key] = np.array(extrinsics[key])
    
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    fake_scenario_path = FAKE_CALIB_SCENARIO_FILEPATH
    fake_scenario = load_scenario(filename=fake_scenario_path, scenario_name='Demo')
    fake_station = MakeFakeStation(fake_scenario, meshcat)
    fake_plant = fake_station.GetSubsystemByName("plant")
    fake_scene_graph = fake_station.GetSubsystemByName("scene_graph")
    
    triad_length=0.25
    triad_radius=0.01
    triad_opacity=0.5
    for key in extrinsics.keys():
        c_X_w = RigidTransform(extrinsics[key])
        w_X_c = c_X_w.inverse()
        source_id = fake_scene_graph.RegisterSource(f'cam_{key}')
        AddAnchoredTriad(
            source_id=source_id,
            scene_graph=fake_scene_graph,
            length=triad_length,
            radius=triad_radius,
            opacity=triad_opacity,
            X_FT=w_X_c,
            name=f'cam_{key}'
        )
    builder.AddSystem(fake_station)
    AddMultibodyTriad(fake_plant.GetFrameByName("tag_frame"), fake_scene_graph)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(diagram_context)
    input("Done")
    