# -*- coding: utf-8 -*-
import math, random, os, shutil, sys, time
from math import sqrt
import csv
from datetime import datetime
from abaqus import *
from abaqusConstants import *
from caeModules import *
import part, material, section, assembly, step, regionToolset
import visualization

# Add EasyPBC to path
sys.path.insert(8, '/abaqus_plugins/EasyPBC')
import easypbc

baseWorkDir = ""
if not os.path.exists(baseWorkDir):
    os.makedirs(baseWorkDir)
    print(f"Created base directory: {baseWorkDir}")

def generateModel(seed_value=None):
    if seed_value is not None:
        random.seed(seed_value)
    Mdb()
    ConcLength, ConcWidth, ConcThickness = 100, 100, 10
    Coverthickness = 0.2
    dmin = 10
    dmax = 12
    minRatio = 0.15
    maxRatio = 0.45
    AggRatio = random.uniform(minRatio, maxRatio)
    print(f"Target fiber area ratio set to: {AggRatio:.3f}")
    InterfaceThickness = 0.2

    def interact_judgement(points, point):
        x1, y1, r1 = point
        return all(sqrt((x1-x2)**2 + (y1-y2)**2) >= (r1+r2+4*InterfaceThickness) for x2,y2,r2 in points)

    AreaConc = ConcLength * ConcWidth
    AggArea = AreaConc * AggRatio
    points, CenterDatas = [], []
    fiberCount, currentAggArea = 0, 0.0
    maxAttempts = 10000
    
    print("Generating fiber distribution...")
    for i in range(maxAttempts):
        if currentAggArea >= AggArea: break
        
        radius = random.uniform(dmin/2, dmax/2)
        x1 = random.uniform(radius + Coverthickness + InterfaceThickness, 
                          ConcLength - radius - Coverthickness - InterfaceThickness)
        y1 = random.uniform(radius + Coverthickness + InterfaceThickness, 
                          ConcWidth - radius - Coverthickness - InterfaceThickness)
        point = (x1, y1, radius)
        t = Coverthickness
        
        if (x1-t-radius-InterfaceThickness > 0 and x1+radius+t+InterfaceThickness < ConcLength and 
            y1-t-radius-InterfaceThickness > 0 and y1+t+radius+InterfaceThickness < ConcWidth and 
            interact_judgement(points, point)):
            
            points.append(point)
            CenterDatas.append([fiberCount, x1, y1, radius])
            fiberCount += 1
            currentAggArea += math.pi * radius**2
    
    print(f"Generated {fiberCount} fibers with area ratio: {currentAggArea/AreaConc:.3f}")

def buildModel(aggregates, voids, ConcLength, ConcWidth, fiberCount, voidCount, modelName, ConcThickness, InterfaceThickness):
    myModel = mdb.models[modelName]

    fiberSketch = myModel.ConstrainedSketch(name='fiberSketch', sheetSize=200.0)
    for fiber in aggregates:
        x1, y1, radius = fiber[1], fiber[2], fiber[3]
        fiberSketch.CircleByCenterPerimeter(center=(x1, y1), point1=(x1 + radius, y1))
    fiberPart = myModel.Part(name='Fibers', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    fiberPart.BaseSolidExtrude(sketch=fiberSketch, depth=ConcThickness)

    interfaceSketch = myModel.ConstrainedSketch(name='interfaceSketch', sheetSize=200.0)
    for fiber in aggregates:
        x1, y1, radius = fiber[1], fiber[2], fiber[3]
        interfaceRadius = radius + InterfaceThickness
        interfaceSketch.CircleByCenterPerimeter(center=(x1, y1), point1=(x1 + interfaceRadius, y1))
        interfaceSketch.CircleByCenterPerimeter(center=(x1, y1), point1=(x1 + radius, y1))
    interfacePart = myModel.Part(name='PureInterfaces', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    interfacePart.BaseSolidExtrude(sketch=interfaceSketch, depth=ConcThickness)

    matrixSketch = myModel.ConstrainedSketch(name='matrixSketch', sheetSize=200.0)

    matrix_E = random.uniform(2000.0, 4500.0)
    matrix_v = random.uniform(0.3, 0.38)
    myModel.Material(name='Matrix_Material').Elastic(table=((matrix_E, matrix_v),))

    fiber_E1 = random.uniform(190000.0, 260000.0)
    fiber_E2 = fiber_E3 = random.uniform(14000.0, 18000.0)
    fiber_G12 = fiber_G13 = random.uniform(10000.0, 20000.0)
    fiber_G23 = random.uniform(5000.0, 9000.0)
    fiber_v12 = fiber_v13 = random.uniform(0.18, 0.24)
    fiber_v23 = random.uniform(0.18, 0.24)
    myModel.Material(name='Fiber_Material').Elastic(
        type=ENGINEERING_CONSTANTS,
        table=((fiber_E1, fiber_E2, fiber_E3, fiber_v12, fiber_v13, fiber_v23, 
                fiber_G12, fiber_G13, fiber_G23),))

    interface_E1 = random.uniform(50000.0, 140000.0)
    interface_E2 = interface_E3 = random.uniform(5000.0, 11000.0)
    interface_G12 = interface_G13 = random.uniform(3000.0, 10000.0)
    interface_G23 = random.uniform(2000.0, 5000.0)
    interface_v12 = interface_v13 = 0.35
    myModel.Material(name='Interface_Material').Elastic(
        type=ENGINEERING_CONSTANTS,
        table=((interface_E1, interface_E2, interface_E3, interface_v12, interface_v13, 
                interface_v23, interface_G12, interface_G13, interface_G23),))

    myModel.HomogeneousSolidSection(name='Matrix_Section', material='Matrix_Material')
    myModel.HomogeneousSolidSection(name='Fiber_Section', material='Fiber_Material')
    myModel.HomogeneousSolidSection(name='Interface_Section', material='Interface_Material')

    matrixPart.SectionAssignment(region=regionToolset.Region(cells=matrixPart.cells), sectionName='Matrix_Section')
    fiberPart.SectionAssignment(region=regionToolset.Region(cells=fiberPart.cells), sectionName='Fiber_Section')
    interfacePart.SectionAssignment(region=regionToolset.Region(cells=interfacePart.cells), sectionName='Interface_Section')

    assembly = myModel.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    for instName in list(assembly.instances.keys()): del assembly.instances[instName]
    
    matrixInstance = assembly.Instance(name='MatrixWithHoles-1', part=matrixPart, dependent=ON)
    fiberInstance = assembly.Instance(name='Fibers-1', part=fiberPart, dependent=ON)
    interfaceInstance = assembly.Instance(name='PureInterfaces-1', part=interfacePart, dependent=ON)

def runEasyPBC():
    try:
        print("Running EasyPBC analysis...")
        easypbc.feasypbc(
            part='Model-1', 
            inst='CompositePart-1', 
            meshsens=1E-04, 
            CPU=8, 
            E11=True, 
            E22=True, 
            E33=True, 
            G12=True, 
            G13=True, 
            G23=True, 
            onlyPBC=False, 
            CTE=False, 
            intemp=0, 
            fntemp=100
        )
        print("EasyPBC analysis completed successfully")
        return True
    except Exception as e:
        print(f"Error in EasyPBC analysis: {str(e)}")
        return False

def copyResultsToFolder(model_index):
    target_dir = os.path.join(baseWorkDir, str(model_index))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    current_dir = os.getcwd()
    files_to_copy = []

    for filename in os.listdir(current_dir):
        if any(filename.endswith(ext) for ext in ['.odb', '.cae', '.inp', '.log', '.dat', '.msg', '.sim', '.sta', '.prt']):
            files_to_copy.append(filename)
    
    return True

def processModel(model_index):
    print(f"\n{'='*50}")
    print(f"Processing model #{model_index}")
    print(f"{'='*50}")

    model_dir = os.path.join(baseWorkDir, str(model_index))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory for model #{model_index}: {model_dir}")

    os.chdir(model_dir)
    print(f"Working directory changed to: {model_dir}")

    aggregates, voids, ConcLength, ConcWidth, fiberCount, voidCount, modelName, ConcThickness, InterfaceThickness = generateModel(seed_value=model_index)
    
    if fiberCount > 0:

        model_built = buildModel(aggregates, voids, ConcLength, ConcWidth, fiberCount, voidCount, modelName, ConcThickness, InterfaceThickness)
        
        if model_built:

            model_file = os.path.join(model_dir, f"Circle_{model_index}.cae")
            mdb.saveAs(model_file)
            print(f"Model saved to {model_file}")

            image_saved = save2DImage(model_dir, model_index)
            if not image_saved:
                print("Warning: Failed to save 2D cross-section image")

            analysis_success = runEasyPBC()
            
            return analysis_success
        else:
            print(f"Failed to build model #{model_index}")
            return False
    else:
        print(f"Error: No fibers were generated for model #{model_index}. Please check your parameters.")
        return False

def batchProcess():
    success_count = 0
    failed_models = []
    
    start_time = time.time()
    
    for i in range(START_INDEX, END_INDEX + 1):
        try:
            success = processModel(i)
            
            if success:
                success_count += 1
            else:
                failed_models.append(i)
            
            if i % SAVE_INTERVAL == 0:

                os.chdir(baseWorkDir)
                mdb.save()
                print(f"Session saved after processing {i} models")

            elapsed_time = time.time() - start_time
            models_done = i - START_INDEX + 1
            avg_time_per_model = elapsed_time / models_done
            remaining_models = END_INDEX - i
            est_time_remaining = avg_time_per_model * remaining_models
            
            print(f"Progress: {models_done}/{END_INDEX-START_INDEX+1} models processed")
            print(f"Average time per model: {avg_time_per_model:.1f} seconds")
            print(f"Estimated time remaining: {est_time_remaining/60:.1f} minutes ({est_time_remaining/3600:.2f} hours)")
            
        except Exception as e:
            print(f"Fatal error processing model #{i}: {str(e)}")
            failed_models.append(i)

    os.chdir(baseWorkDir)

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"Batch processing complete!")
    print(f"Processed {END_INDEX-START_INDEX+1} models in {total_time/3600:.2f} hours")
    
    if failed_models:
        print("Failed model indices:", failed_models)
    
    print("="*50)

if __name__ == "__main__":
    print("Starting batch processing of fiber composite models...")
    batchProcess()