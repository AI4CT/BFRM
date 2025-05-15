from pxr import Usd, UsdGeom, Gf, Vt, Sdf
import pyvista as pv
import numpy as np
import matplotlib as cm
import time

def calculate_color(mesh, variable, min, max):
    
    print(mesh.cell_data)
    scalars = mesh.cell_data[variable]

    if(scalars.ndim==1):
        scalars_normalized = (scalars - min) / (max - min)

    elif(scalars.ndim==2):
        magnitudes = np.linalg.norm(scalars, axis=1)
        scalars_normalized = (magnitudes - min) / (max - min)
        
    colormap = cm.colormaps.get_cmap("jet")
    return colormap(scalars_normalized)[:, :3]

def main(usd_path, time_length, variable, min, max):
    
    stage = Usd.Stage.CreateNew(usd_path)
    default_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(default_prim.GetPrim())

    for i in range(1, time_length+1):
        vtk_path = f"./vtk_file/Bumper_Beam_AP_meshedA{i:03d}.vtk"
        output = pv.read(vtk_path)
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        time_code_1 = Usd.TimeCode((i - 1))  
        
        points = output.points.astype(np.float32)
        mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points), time_code_1)

        if i ==1:
            face = output.cells.tolist()
            face_vertex_counts = []
            face_vertex_indices = []

            m = 0
            while m < len(face):
                n_points = face[m]
                face_vertex_counts.append(n_points) 
                face_vertex_indices.extend(face[m+1:m + n_points + 1])
                m += n_points + 1
    
            mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
            mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    

        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        colors = calculate_color(output, variable, min, max)

        color_primvar = primvars_api.CreatePrimvar(
                "displayColor",
                Sdf.ValueTypeNames.Color3fArray,
                interpolation="uniform"
        )

        color_primvar.Set([Gf.Vec3f(*c) for c in colors], time_code_1)
        print(f"{i} success")
        
    stage.Save()
    print("succes!")   

if __name__ == "__main__":
    
    usd_path = "output.usda"
    time_length = 10
    variable = "2DELEM_Von_Mises"
    min = 0
    max = 0.5

    start_time = time.process_time()
    main(usd_path, time_length, variable, min, max)
    end_time = time.process_time()
    
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
