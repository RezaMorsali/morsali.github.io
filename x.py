# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 6.14-1 replay file
# Internal Version: 2014_06_04-17.11.02 134264
# Run by sxm150632 on Fri May 03 10:15:16 2019
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
#: Warning: Permission was denied for "abaqus.rpy"; "abaqus.rpy.2" will be used for this session's replay file.
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=50.0, 
    height=50.0)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
#: Executing "onCaeStartup()" in the site directory ...
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
Mdb()
#: A new model database has been created.
#: The model "Model-1" has been created.
# read the file
ABQinfo=open('ABQinfo.txt','r')
info_test=[]
for i in ABQinfo:
 info_test.append(i.strip("\n"))
LR=float(info_test[0])
WR=float(info_test[1])
AR=float(info_test[2])
w=float(info_test[3])
h=float(info_test[4])
tone=float(info_test[5])
ttwo=float(info_test[6])
sh=float(info_test[7])
Ebrick=float(info_test[8])
Emortar=float(info_test[9])
Pbrick=float(info_test[10])
Pmortar=float(info_test[11])
################## Geometry
session.viewports['Viewport: 1'].setValues(displayedObject=None)
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=(0.0, 0.0), point2=(LR, WR))
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=TWO_D_PLANAR, 
    type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-1']
p.BaseShell(sketch=s)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['Model-1'].parts['Part-1']
f, e, d1 = p.faces, p.edges, p.datums
t = p.MakeSketchTransform(sketchPlane=f[0], sketchPlaneSide=SIDE1, origin=(0.0, 
    0.0, 0.0))
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2.82, 
    gridSpacing=0.07, transform=t)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=SUPERIMPOSE)
p = mdb.models['Model-1'].parts['Part-1']
p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
s.rectangle(point1=(0, ttwo/2), point2=(LR, h+ttwo/2))
s.rectangle(point1=(tone/2, 0), point2=(tone/2+w, WR))
s.rectangle(point1=(0, 3*ttwo/2+h), point2=(LR, 3*ttwo/2+2*h))
s.rectangle(point1=(w-sh, 0), point2=(w-sh+tone, WR))
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
pickedFaces = f.getSequenceFromMask(mask=('[#1 ]', ), )
e1, d2 = p.edges, p.datums
p.PartitionFaceBySketch(faces=pickedFaces, sketch=s)
s.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']
########################## properties
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
mdb.models['Model-1'].Material(name='brick')
mdb.models['Model-1'].materials['brick'].Elastic(table=((Ebrick, Pbrick), ))
mdb.models['Model-1'].Material(name='mortar')
mdb.models['Model-1'].materials['mortar'].Elastic(table=((Emortar, Pmortar), ))
mdb.models['Model-1'].HomogeneousSolidSection(name='brick', material='brick', 
    thickness=None)
mdb.models['Model-1'].HomogeneousSolidSection(name='mortar', material='mortar', 
    thickness=None)
session.viewports['Viewport: 1'].setValues(displayedObject=None)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
faces = f.getSequenceFromMask(mask=('[#1828a20 ]', ), )
region = p.Set(faces=faces, name='Set-1')
p = mdb.models['Model-1'].parts['Part-1']
p.SectionAssignment(region=region, sectionName='brick', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
faces = f.getSequenceFromMask(mask=('[#7d75df ]', ), )
region = p.Set(faces=faces, name='Set-2')
p = mdb.models['Model-1'].parts['Part-1']
p.SectionAssignment(region=region, sectionName='mortar', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
##### Asembly
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=OFF)
############################ Step
#: Executing "onCaeStartup()" in the site directory ...
#: A new model database has been created.
#: The model "Model-1" has been created.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
mdb.models['Model-1'].StaticStep(name='staticload', previous='Initial')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='staticload')
############### interaction

session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    adaptiveMeshConstraints=OFF)
a = mdb.models['Model-1'].rootAssembly
v1 = a.instances['Part-1-1'].vertices
a.ReferencePoint(point=v1[10])
a = mdb.models['Model-1'].rootAssembly
r1 = a.referencePoints
refPoints1=(r1[4], )
a.Set(referencePoints=refPoints1, name='Set_ref')
#: The set 'Set_ref' has been created (1 reference point).
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
edges1 = e1.getSequenceFromMask(mask=('[#20004814 ]', ), )
a.Set(edges=edges1, name='set_x')
#: The set 'set_x' has been created (5 edges).
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
edges1 = e1.getSequenceFromMask(mask=('[#4001000 #410100 ]', ), )
a.Set(edges=edges1, name='set_y')
#: The set 'set_y' has been created (5 edges).
mdb.models['Model-1'].Equation(name='eqx', terms=((1.0, 'set_x', 1), (-1.0, 
    'Set_ref', 1)))
mdb.models['Model-1'].Equation(name='eqy', terms=((1.0, 'set_y', 2), (-1.0, 
    'Set_ref', 2)))
##############load
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
edges1 = e1.getSequenceFromMask(mask=('[#1200202 #10 ]', ), )
region = a.Set(edges=edges1, name='Set-4')
mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Initial', 
    region=region, u1=UNSET, u2=SET, ur3=UNSET, amplitude=UNSET, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
edges1 = e1.getSequenceFromMask(mask=('[#800000 #8220080 ]', ), )
region = a.Set(edges=edges1, name='Set-5')
mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Initial', 
    region=region, u1=SET, u2=UNSET, ur3=UNSET, amplitude=UNSET, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='staticload')
a = mdb.models['Model-1'].rootAssembly
s1 = a.instances['Part-1-1'].edges
side1Edges1 = s1.getSequenceFromMask(mask=('[#20004814 ]', ), )
region = a.Surface(side1Edges=side1Edges1, name='Surf-1')
mdb.models['Model-1'].Pressure(name='tension', createStepName='staticload', 
    region=region, distributionType=UNIFORM, field='', magnitude=-1.0, 
    amplitude=UNSET)
#############Mesh
# change the numbers below to decrease or increase the seeds in model
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
    bcs=OFF, predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
pickedEdges = e1.getSequenceFromMask(mask=('[#bd0bd2f #178240 ]', ), )
a.seedEdgeByNumber(edges=pickedEdges, number=1, constraint=FINER)
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
pickedEdges = e1.getSequenceFromMask(mask=('[#942d4280 #283505 ]', ), )
a.seedEdgeByNumber(edges=pickedEdges, number=2, constraint=FINER)
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
pickedEdges = e1.getSequenceFromMask(mask=('[#0 #2c04018 ]', ), )
a.seedEdgeByNumber(edges=pickedEdges, number=5, constraint=FINER)
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
pickedEdges = e1.getSequenceFromMask(mask=('[#60020050 #d0008a2 ]', ), )
a.seedEdgeByNumber(edges=pickedEdges, number=4, constraint=FINER)
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
pickedRegions = f1.getSequenceFromMask(mask=('[#1ffffff ]', ), )
a.setMeshControls(regions=pickedRegions, technique=STRUCTURED)
a = mdb.models['Model-1'].rootAssembly
partInstances =(a.instances['Part-1-1'], )
a.generateMesh(regions=partInstances)
elemType1 = mesh.ElemType(elemCode=CPE4R, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, hourglassControl=DEFAULT, 
    distortionControl=DEFAULT)
elemType2 = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, distortionControl=DEFAULT)
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#1ffffff ]', ), )
pickedRegions =(faces1, )
a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
# creating extra sets
#: The set 'brick' has been created (72 elements).
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON)
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].elements
elements1 = e1.getSequenceFromMask(mask=(
    '[#8783fc00 #c3fc007f #3ffff #fffff000 #f ]', ), )
a.Set(elements=elements1, name='brick')
## Vertical interface
#: The set 'VI' has been created (16 elements).
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON)
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].elements
elements2 = e1.getSequenceFromMask(mask=('[#1e #f000 #fc000000 #3 ]', ), )
a.Set(elements=elements2, name='VI')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
# Horizantal interface
session.viewports['Viewport: 1'].setValues(displayedObject=p)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON)
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].elements
elements3 = e1.getSequenceFromMask(mask=(
    '[#787c03e1 #3c030f80 #3fc0000 #ffc ]', ), )
a.Set(elements=elements3, name='HI')
####### job creation
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
mdb.Job(name='Job-1brick', model='Model-1', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
    numGPUs=0)
###### job submission
mdb.jobs['Job-1brick'].submit(consistencyChecking=OFF)
#: The job input file "Job-1brick.inp" has been submitted for analysis.
#: Job Job-1brick: Analysis Input File Processor completed successfully.
#: Job Job-1brick: Abaqus/Standard completed successfully.
#: Job Job-1brick completed successfully. 
#====================================================================================
from odbAccess import *
from abaqusConstants import *
import operator
odb=openOdb('Job-1brick.odb')
dispFile=open('result.txt','w')
lastFrame = odb.steps['staticload'].frames[-1]
displacement=lastFrame.fieldOutputs['U']
fieldValues=displacement.values
for v in fieldValues: #young modulus
	if ((v.nodeLabel==11)):
		#for i in range(1):
		dispFile.write((str(1/v.data[0])+' '))
		dispFile.write(('\n'))
dispField = lastFrame.fieldOutputs['S']
fieldValues=dispField.values
Sx=[]
for v in fieldValues: #maximum principal stress in all the elements
		Sx.append(v.data[0])
MaxSx=max(Sx)
index, value = max(enumerate(Sx), key=operator.itemgetter(1))
Elable=dispField.values[index].elementLabel
dispFile.write((str(Elable)+' '+str(MaxSx)))
dispFile.write(('\n'))
# max smises in horizental interface
centerNSet = odb.rootAssembly.elementSets['HI']
dispSubField = dispField.getSubset(region=centerNSet)
fieldValues=dispSubField.values
Sx=[]
for v in fieldValues: #maximum yield stress in horizental interface
		Sx.append(v.mises)
MaxSx=max(Sx)
index, value = max(enumerate(Sx), key=operator.itemgetter(1))
Elable=dispField.values[index].elementLabel
dispFile.write((str(Elable)+' '+str(MaxSx)))
dispFile.write(('\n'))
# max smises in vertical interface
centerNSet = odb.rootAssembly.elementSets['VI']
dispSubField = dispField.getSubset(region=centerNSet)
fieldValues=dispSubField.values
Sx=[]
for v in fieldValues: #maximum yield stress in vertical interface
		Sx.append(v.mises)
MaxSx=max(Sx)
index, value = max(enumerate(Sx), key=operator.itemgetter(1))
Elable=dispField.values[index].elementLabel
dispFile.write((str(Elable)+' '+str(MaxSx)))
dispFile.close()
odb.close()






