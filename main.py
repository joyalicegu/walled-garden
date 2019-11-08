from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.showbase import Audio3DManager
from direct.interval.IntervalGlobal import *
from direct.task.Task import Task
from direct.gui.DirectGui import *
import sys, random, math, collections
from functools import reduce

# OH NO! IT'S A GLOBAL VARIABLE.
DIRECTIONS = [(-1,  0,  0), (+1,  0,  0),
              ( 0, -1,  0), ( 0, +1,  0),
              ( 0,  0, -1), ( 0,  0, +1)]
# OH NO! ANOTHER ONE!
PHI = (1 + math.sqrt(5))/2.0

def mazeGenerator(rows, cols, lays):
    # Returns a set of frozensets of 3-tuples.
    # Orthogonal perfect maze.
    # Maximum recursion depth of this is the number of cells.
    # Create adjacency set of walls (walls everywhere).
    # We have also have walls between, for example, (0, 0, 0) and (0, -1, 0).
    walls = set()
    for row in range(rows):
        for col in range(cols):
            for lay in range(lays):
                if row == 0:
                    walls.add(frozenset([(row, col, lay), (row-1, col, lay)]))
                if col == 0:
                    walls.add(frozenset([(row, col, lay), (row, col-1, lay)]))
                if lay == 0:
                    walls.add(frozenset([(row, col, lay), (row, col, lay-1)]))
                walls.add(frozenset([(row, col, lay), (row+1, col, lay)]))
                walls.add(frozenset([(row, col, lay), (row, col+1, lay)]))
                walls.add(frozenset([(row, col, lay), (row, col, lay+1)]))
    visited = set()
    def removeWalls(row, col, lay):
        visited.add((row, col, lay))
        # In a random order of DIRECTIONS,
        for drow, dcol, dlay in random.sample(DIRECTIONS, len(DIRECTIONS)):
            nrow, ncol, nlay = row + drow, col + dcol, lay + dlay
            # If new cell is in the maze and not visited,
            if ((0 <= nrow < rows and 0 <= ncol < cols and 0 <= nlay < lays)
                    and ((nrow, ncol, nlay) not in visited)):
                # Then break down the wall!
                walls.discard(frozenset([(row, col, lay), (nrow, ncol, nlay)]))
                # Call removeWalls on the new cell.
                removeWalls(nrow, ncol, nlay)
        # Base case: dead end.
    removeWalls(0, 0, 0)
    return walls

def neighbors(walls, rows, cols, lays, cell):
    (row, col, lay) = cell
    nextCells = list()
    for (drow, dcol, dlay) in DIRECTIONS:
        nextCell = (row + drow, col + dcol, lay + dlay)
        if ((frozenset([nextCell, cell]) not in walls) and
                (0 <= nextCell[0] < rows) and (0 <= nextCell[1] < cols) and
                (0 <= nextCell[2] < lays)):
            nextCells.append(nextCell)
    return nextCells

def deadEnds(walls, rows, cols, lays):
    deadEnds = set()
    cells = [(row, col, lay) for row in range(rows)
             for col in range(cols) for lay in range(lays)]
    for cell in cells:
        nextCells = neighbors(walls, rows, cols, lays, cell)
        if len(nextCells) == 1:
            deadEnds.add(cell)
    return deadEnds

def cellDistancesFromSource(walls, rows, cols, lays, source):
    currentCells = collections.deque()
    currentCells.append(source)
    visited = set()
    visited.add(source)
    dist = dict()
    dist[source] = 0
    while len(currentCells) != 0:
        currentCell = currentCells.popleft()
        nextCells = neighbors(walls, rows, cols, lays, currentCell)
        for nextCell in nextCells:
            if nextCell not in visited:
                visited.add(nextCell)
                currentCells.append(nextCell)
                dist[nextCell] = dist[currentCell] + 1
    return dist

def pathToNearestTarget(walls, rows, cols, lays, source, targets):
    targets = set(targets)
    currentCells = collections.deque()
    currentCells.append(source)
    visited = set()
    visited.add(source)
    path = dict()
    path[source] = [source]
    while len(currentCells) != 0:
        currentCell = currentCells.popleft()
        if currentCell in targets:
            return path[currentCell]
        nextCells = neighbors(walls, rows, cols, lays, currentCell)
        for nextCell in nextCells:
            if nextCell not in visited:
                visited.add(nextCell)
                currentCells.append(nextCell)
                path[nextCell] = path[currentCell] + [nextCell]

def hsvToRgb(h, s, v):
    # Formula from https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
    x = (1 - abs(h/60.0 % 2 - 1))
    r, g, b = [(1, x, 0), # red to yellow (more red than green)
               (x, 1, 0), # yellow to green (more green than red)
               (0, 1, x), # green to cyan (more green than blue)
               (0, x, 1), # cyan to blue (more blue than green)
               (x, 0, 1), # blue to purple (more blue than red)
               (1, 0, x)  # purple to red (more red than blue)
               ][h//60] 
    return tuple([v*s*(x-1)+v for x in [r, g, b]])
    # r, g, b are floats from 0 to 1 inclusive

def keyColorsFromScores(scores):
    minScore, maxScore = min(scores.values()), max(scores.values())
    minHue, maxHue = 0, 300 # red to purple
    step = (maxHue-minHue) / float(maxScore-minScore)
    colors = dict()
    alpha = 1
    for key in scores:
        hue = int(minHue + (scores[key] - minScore) * step)
        (r, g, b) = hsvToRgb(hue, 1.0, 1.0)
        colors[key] = (r, g, b, alpha)
    return colors

def keyGraysFromScores(scores):
    minScore, maxScore = min(scores.values()), max(scores.values())
    minGray, maxGray = 0.9, 0.5
    step = (maxGray-minGray) / float(maxScore-minScore)
    grays = dict()
    alpha = 1
    for key in scores:
        gray = minGray + (scores[key] - minScore) * step
        grays[key] = (gray, gray, gray, alpha)
    return grays

def norm(face):
    x0, y0, z0 = face[0]
    x1, y1, z1 = face[1]
    x2, y2, z2 = face[2]
    vector1 = LVecBase3f(x1-x0, y1-y0, z1-z0)
    vector2 = LVecBase3f(x2-x0, y2-y0, z2-z0)
    return vector1.cross(vector2)

def tetrahedron(size=1):
    size = size/2.0
    aa = (    0,                    0, size*3/math.sqrt(6))
    bb = ( size,   -size/math.sqrt(3),  -size/math.sqrt(6))
    cc = (-size,   -size/math.sqrt(3),  -size/math.sqrt(6))
    dd = (    0,  size*2/math.sqrt(3),  -size/math.sqrt(6))
    format = GeomVertexFormat.getV3n3c4()
    vertexData = GeomVertexData('tetrahedron', format, Geom.UHStatic)
    vertexData.setNumRows(12)
    vertices = GeomVertexWriter(vertexData, 'vertex')
    normals = GeomVertexWriter(vertexData, 'normal')
    primitive = GeomTriangles(Geom.UHStatic)
    # Vertices are counterclockwise from front.
    faces = [[dd, bb, cc], [dd, aa, bb], [bb, aa, cc], [cc, aa, dd]]
    norms = [norm(face) for face in faces]
    for face in range(4):
        # Add vertices.
        for point in faces[face]:
            vertices.addData3f(*point)
        # Add normals.
        for i in range(3): normals.addData3f(*norms[face])
        # Add face to primitive.
        primitive.add_consecutive_vertices(face*3, 3)
        primitive.close_primitive()
    geom = Geom(vertexData)
    geom.addPrimitive(primitive)
    return geom

def sierpinskiTetrahedron(level=4, size=1):
    if level > 4: level = 4
    if (level<=0):
        # tetrahedron is centered at (0, 0, 0), with side length = 2
        result = NodePath(GeomNode('tetrahedron'))
        result.node().addGeom(tetrahedron(2))
        result.setScale(size)
    else:
        size = size/2.0
        result = NodePath(str(level))
        st = sierpinskiTetrahedron(level-1, size)
        aa, bb, cc, dd = (st.copyTo(result) for i in range(4))
        aa.setPos(    0,                    0, size*3/math.sqrt(6))
        bb.setPos( size,   -size/math.sqrt(3),  -size/math.sqrt(6))
        cc.setPos(-size,   -size/math.sqrt(3),  -size/math.sqrt(6))
        dd.setPos(    0,  size*2/math.sqrt(3),  -size/math.sqrt(6))
        result.flattenStrong()
    return result

def kochSnowflake(level=4, size=1):
    if level > 4: level = 4
    if (level<=0):
        # tetrahedron is centered at (0, 0, 0), with side length = 2
        result = NodePath(GeomNode('tetrahedron'))
        result.node().addGeom(tetrahedron(2))
        result.setScale(size)
    else:
        result = NodePath(str(level))
        body = kochSnowflake(0, size)
        body.copyTo(result)
        aa, bb, cc, dd = (result.attachNewNode("pivot") for i in range(4))
        pitch = math.degrees(math.acos(1/3.0)) + 180
        aa.setHpr(120, pitch, 180)
        bb.setHpr(240, pitch, 180)
        cc.setHpr(  0, pitch, 180)
        dd.setHpr(0, 180, 0)
        spike = kochSnowflake(level-1, size/2.0)
        for pivot in (aa, bb, cc, dd):
            copy = spike.copyTo(pivot)
            height = 1/math.sqrt(6)
            copy.setPos(0, 0, 1.5*size*height)
        result.flattenStrong()
    return result

def dodecahedron(size=1):
    pos = ([(x, y, z) for x in [-PHI, PHI]
                 for y in [-PHI, PHI] for z in [-PHI, PHI]]
              + [(0, y, z) for y in [-1, 1] for z in [-PHI**2, PHI**2]]
              + [(x, 0, z) for x in [-PHI**2, PHI**2] for z in [-1, 1]]
              + [(x, y, 0) for x in [-1, 1] for y in [-PHI**2, PHI**2]])
    format = GeomVertexFormat.getV3n3c4()
    vertexData = GeomVertexData('dodecahedron', format, Geom.UHStatic)
    vertexData.setNumRows(60)
    vertices = GeomVertexWriter(vertexData, 'vertex')
    normals = GeomVertexWriter(vertexData, 'normal')
    primitive = GeomTrifans(Geom.UHStatic)
    # Vertices are counterclockwise from front.
    faces = [[17, 19, 6, 10, 2],
             [14, 15, 5, 18, 4],
             [0, 16, 1, 13, 12],

             [2, 10, 8, 0, 12],
             [10, 6, 14, 4, 8],
             [8, 4, 18, 16, 0],

             [17, 3, 11, 7, 19],
             [3, 13, 1, 9, 11],
             [7, 11, 9, 5, 15],

             [2, 12, 13, 3, 17],
             [19, 7, 15, 14, 6],
             [1, 16, 18, 5, 9]]
    faces = [[pos[i] for i in face] for face in faces]
    norms = [norm(face) for face in faces]
    for face in range(12):
        # Add vertices.
        for point in faces[face]:
            vertices.addData3f(*[x*size/2.0 for x in point])
        # Add normals.
        for i in range(5): normals.addData3f(*norms[face])
        # Add face to primitive.
        primitive.add_consecutive_vertices(face*5, 5)
        primitive.close_primitive()
    geom = Geom(vertexData)
    geom.addPrimitive(primitive)
    return geom

def sierpinskiDodecahedron(level=2, size=1):
    if level > 2: level = 2
    if (level<=0):
        # Dodecahedron is centered at (0, 0, 0), with edge length = 1
        result = NodePath(GeomNode('dodecahedron'))
        result.node().addGeom(dodecahedron(1))
        result.setScale(size)
    else:
        result = NodePath(str(level))
        sd = sierpinskiDodecahedron(level-1, size/PHI**2)
        center = [sd.copyTo(result) for i in range(20)]
        position = ([(x, y, z) for x in [-PHI, PHI]
                     for y in [-PHI, PHI] for z in [-PHI, PHI]]
                  + [(0, y, z) for y in [-1, 1] for z in [-PHI**2, PHI**2]]
                  + [(x, 0, z) for x in [-PHI**2, PHI**2] for z in [-1, 1]]
                  + [(x, y, 0) for x in [-1, 1] for y in [-PHI**2, PHI**2]])
        for i in range(20):
            center[i].setPos(*[x*size/2.0 for x in position[i]])
        result.setScale(PHI/2)
        result.flattenStrong()
    return result

def icosahedron(size=1):
    pos = ([(0, y, z) for y in [-PHI, PHI] for z in [-1, 1]]
         + [(x, 0, z) for z in [-PHI, PHI] for x in [-1, 1]]
         + [(x, y, 0) for x in [-PHI, PHI] for y in [-1, 1]])
    format = GeomVertexFormat.getV3n3c4()
    vertexData = GeomVertexData('icosahedron', format, Geom.UHStatic)
    vertexData.setNumRows(36)
    vertices = GeomVertexWriter(vertexData, 'vertex')
    normals = GeomVertexWriter(vertexData, 'normal')
    primitive = GeomTrifans(Geom.UHStatic)
    # Vertices are counterclockwise from front.
    faces = [[9, 6, 3], [9, 3, 2], [5, 11, 10], [5, 2, 11], [7, 1, 10],
             [7, 6, 1], [2, 3, 11], [6, 7, 3], [10, 11, 7], [3, 7, 11],
             [2, 4, 9], [2, 5, 4], [8, 6, 9], [8, 1, 6], [0, 10, 1],
             [0, 5, 10], [9, 4, 8], [5, 0, 4], [1, 8, 0], [4, 0, 8]]
    faces = [[pos[i] for i in face] for face in faces]
    norms = [norm(face) for face in faces]
    for face in range(20):
        # Add vertices.
        for point in faces[face]:
            vertices.addData3f(*[x*size/2.0 for x in point])
        # Add normals.
        for i in range(3): normals.addData3f(*norms[face])
        # Add face to primitive.
        primitive.add_consecutive_vertices(face*3, 3)
        primitive.close_primitive()
    geom = Geom(vertexData)
    geom.addPrimitive(primitive)
    return geom

def sierpinskiIcosahedron(level=2, size=2):
    if level > 2: level = 2
    if (level<=0):
        # Icosahedron is centered at (0, 0, 0), with edge length = 1
        result = NodePath(GeomNode('Icosahedron'))
        result.node().addGeom(icosahedron(1))
        result.setScale(size)
    else:
        result = NodePath(str(level))
        si = sierpinskiIcosahedron(level-1, size/PHI**2)
        center = [si.copyTo(result) for i in range(12)]
        position = ([(0, y, z) for y in [-PHI, PHI] for z in [-1, 1]]
                  + [(x, 0, z) for z in [-PHI, PHI] for x in [-1, 1]]
                  + [(x, y, 0) for x in [-PHI, PHI] for y in [-1, 1]])
        for i in range(12):
            center[i].setPos(*[x*size/(PHI*2) for x in position[i]])
        result.flattenStrong()
    return result

def cubeFlake(level=3, size=1):
    if level > 3: level = 3
    if (level<=0):
        # Cube of side-length 2, centered at (0, 0, 0)
        result = NodePath(GeomNode('cube'))
        result.node().addGeom(cubeFromCorners((-1, -1, -1), (1, 1, 1)))
        result.setScale(size)
    else:
        result = NodePath(str(level))
        body = cubeFlake(0, size)
        body.copyTo(result)
        size = size/3.0
        limb = cubeFlake(level-1, size)
        aa, bb, cc, dd, ee, ff = (limb.copyTo(result) for i in range(6))
        aa.setX(+4*size)
        bb.setX(-4*size)
        cc.setY(+4*size)
        dd.setY(-4*size)
        ee.setZ(+4*size)
        ff.setZ(-4*size)
        result.flattenStrong()
    return result

def mengerSponge(level=2, size=1):
    if level > 3: level = 3
    if (level<=0):
        # Cube of side-length 2, centered at (0, 0, 0)
        result = NodePath(GeomNode('cube'))
        result.node().addGeom(cubeFromCorners((-1, -1, -1), (1, 1, 1)))
        result.setScale(size)
    else:
        result = NodePath(str(level))
        size = size/3.0
        proto = mengerSponge(level-1, size)
        ms = dict()
        for key in [(x, y, z) for x in range(-1, 2)
                for y in range(-1, 2) for z in range(-1, 2)
                if (x, y, z).count(0) < 2]:
            ms[key] = proto.copyTo(result)
        for key in ms:
            ms[key].setPos(*[2*size*x for x in key])
        result.flattenStrong()
    return result

def vicsekFractal(level=3, size=1):
    if level > 3: level = 3
    if (level<=0):
        # Cube of side-length 2, centered at (0, 0, 0)
        result = NodePath(GeomNode('cube'))
        result.node().addGeom(cubeFromCorners((-1, -1, -1), (1, 1, 1)))
        result.setScale(size)
    else:
        result = NodePath(str(level))
        size = size/3.0
        proto = vicsekFractal(level-1, size)
        ms = dict()
        for key in [(x, y, z) for x in range(-1, 2)
                for y in range(-1, 2) for z in range(-1, 2)
                if (x, y, z).count(0) in {0, 3}]:
            ms[key] = proto.copyTo(result)
        for key in ms:
            ms[key].setPos(*[2*size*x for x in key])
        result.flattenStrong()
    return result

def pythagorasTree(level=5, size=0.5, angle=30):
    if level > 5: level = 5
    if (level<=0):
        # Cube of side-length 2, centered at (0, 0, 0)
        result = NodePath(GeomNode('cube'))
        result.node().addGeom(cubeFromCorners((-1, -1, -1), (1, 1, 1)))
        result.setScale(size)
    else:
        result = NodePath(str(level))
        body = pythagorasTree(0, size)
        body.copyTo(result)
        limbs = {angle: result.attachNewNode("pivot"),
                 90-angle: result.attachNewNode("pivot")}
        limbs[angle].setPos(size, size, size)
        limbs[90-angle].setPos(-size, -size, size)
        limbs[angle].setHpr(180-45, angle, 0)
        limbs[90-angle].setHpr(-45, 90-angle, 0)

        for key in limbs:
            length = size*math.cos(math.radians(key))
            tree = pythagorasTree(level-1, length, angle)
            copy = tree.copyTo(limbs[key])
            copy.setPos(0, length*math.sqrt(2), length)
            copy.setHpr(180+45, 0, 0)
        result.flattenStrong()
    return result

def pythagorasFlake(level=5, size=0.2, angle=30):
    if level > 5: level = 5
    pt = pythagorasTree(level, size, angle)
    result = NodePath(str('pycube'))
    pivots = [result.attachNewNode("pivot") for i in range(6)]
    for pivot in pivots: pt.copyTo(pivot)
    pivots[1].setHpr(90,  0, 180)
    pivots[2].setHpr(0, -90, 90)
    pivots[3].setHpr(0, 0, -90)
    pivots[4].setHpr(90,0,-90)
    pivots[5].setHpr(-90,-90,-90)
    result.flattenStrong()
    return result

def scale(a, scalar):
    return [x*scalar for x in a]

def cubeCornersFromWall(wall, cellSize, thickness):
    rows = [cell[0] for cell in wall]
    cols = [cell[1] for cell in wall]
    lays = [cell[2] for cell in wall]

    if cols[0] != cols[1]:
        x0, y0, z0 = max(cols), rows[0]  , lays[0]
        x1, y1, z1 = max(cols), rows[0]+1, lays[0]+1
    elif rows[0] != rows[1]:
        x0, y0, z0 = cols[0]  , max(rows), lays[0]
        x1, y1, z1 = cols[0]+1, max(rows), lays[0]+1
    elif lays[0] != lays[1]:
        x0, y0, z0 = cols[0]  , rows[0]  , max(lays)
        x1, y1, z1 = cols[0]+1, rows[0]+1, max(lays)

    x0, y0, z0, x1, y1, z1 = scale((x0, y0, z0, x1, y1, z1), cellSize)

    return ((x0-thickness, y0-thickness, z0-thickness),
            (x1+thickness, y1+thickness, z1+thickness))

def polygonCornersFromWall(wall, cellSize):
    rows = [cell[0] for cell in wall]
    cols = [cell[1] for cell in wall]
    lays = [cell[2] for cell in wall]

    if cols[0] != cols[1]:
        x, y0, z0, y1, z1 = scale((max(cols), rows[0], lays[0], rows[0]+1, lays[0]+1), cellSize)
        return ((x, y1, z0), (x, y0, z0), (x, y0, z1), (x, y1, z1))
    elif rows[0] != rows[1]:
        y, x0, z0, x1, z1 = scale((max(rows), cols[0], lays[0], cols[0]+1, lays[0]+1), cellSize)
        return ((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))
    elif lays[0] != lays[1]:
        z, x0, y0, x1, y1 = scale((max(lays), cols[0], rows[0], cols[0]+1, rows[0]+1), cellSize)
        return ((x0, y0, z), (x0, y1, z), (x1, y1, z), (x1, y0, z))

def getCellFromPoint(cellSize, x, y, z):
    return tuple([int(x//cellSize) for x in [y, x, z]])

def getCellCenter(cellSize, row, col, lay):
    return tuple([cellSize * (x + 0.5) for x in [col, row, lay]])

def cubeFromCorners(corner0, corner1, cellColorDict=None,
        cellSize=None):
    (x0, y0, z0) = corner0
    (x1, y1, z1) = corner1
    format = GeomVertexFormat.getV3n3c4()
    vertexData = GeomVertexData('cube', format, Geom.UHStatic)
    vertexData.setNumRows(24)

    vertices = GeomVertexWriter(vertexData, 'vertex')
    normals = GeomVertexWriter(vertexData, 'normal')
    if cellColorDict != None:
        colors = GeomVertexWriter(vertexData, 'color')
    primitive = GeomTrifans(Geom.UHStatic)

    # Create vertices, then add each face to primitive.
    # Add each face to primitive.
    # Vertices are counterclockwise from the front.

    faces = [[(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
             [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], 
             [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
             [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
             [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],
             [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]

    norms = [(-1, 0, 0), (1, 0, 0),
             ( 0,-1, 0), (0, 1, 0),
             ( 0, 0,-1), (0, 0, 1)]

    if cellColorDict != None:
        faceColorDict = dict()
        for face in range(6):
            points = faces[face]
            avgX = sum([point[0] for point in points]) / float(len(points))
            avgY = sum([point[1] for point in points]) / float(len(points))
            avgZ = sum([point[2] for point in points]) / float(len(points))
            avgPoint = (avgX, avgY, avgZ)
            cell = getCellFromPoint(cellSize, *avgPoint)
            if cell not in cellColorDict:
                faceColorDict[face] = (1, 1, 1, 1)
            else:
                faceColorDict[face] = cellColorDict[cell]

    for face in range(len(faces)):
        for point in faces[face]:
            # Add vertices.
            vertices.addData3f(*point)
            # Add normals.
            normals.addData3f(*norms[face])
            # Add colors.
            if cellColorDict != None: colors.addData4f(*faceColorDict[face])
        # Add face to primitive.
        vertexCount = 4
        primitive.add_consecutive_vertices(face*vertexCount, vertexCount)
        primitive.close_primitive()

    geom = Geom(vertexData)
    geom.addPrimitive(primitive)
    return geom

def addCollisionPolygonsFromCorners(corner0, corner1, node):
    # Vertices are counterclockwise from the front.
    (x0, y0, z0) = corner0
    (x1, y1, z1) = corner1
    faces = [[(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
             [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], 
             [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
             [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
             [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],
             [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]
    for face in faces:
        points = [Point3(*point) for point in face]
        node.addSolid(CollisionPolygon(*points))

def onscreenMessage(line, msg):
    lineHeight = 0.09
    spacing = 0.01
    return OnscreenText(text=msg, scale=lineHeight,
                        fg=(1,1,1,1), shadow=(0,0,0,0.5),
                        parent=base.a2dTopLeft,
                        pos=(0.04, -line*(lineHeight+spacing)-0.02),
                        align=TextNode.ALeft)

class Flower(object):
    unbloomed = set()
    def __init__(self, cell):
        Flower.unbloomed.add(cell)

class MazeExplorer(ShowBase):

    def __init__(self, rows=5, cols=5, lays=5, mouselook=False):

        ShowBase.__init__(self)

        self.mouselook = mouselook
        self.rows, self.cols, self.lays = rows, cols, lays

        # Call all the init helpers, in order.
        self.initMaze()
        self.initVisibleGeometry()
        self.initCameraLight()
        self.initCollisionGeometry()
        self.initCollisionDetection()
        self.initAudioManager()
        self.initStereo()
        self.initCameraControl()
        self.initBloomTask()
        self.initControls()
        self.initInst()
        self.initWinProps()

    def initWinProps(self):
        self.winProps = WindowProperties()
        self.winProps.setTitle("Walled Garden")
        self.winProps.setCursorHidden(True)
        self.win.requestProperties(self.winProps)
        self.accept("escape", sys.exit, [0])

    def initMaze(self):
        # Limit size of maze.
        maxMazeSize = sys.getrecursionlimit()
        if self.rows*self.cols*self.lays > maxMazeSize:
            self.rows = self.cols = self.lays = int(maxMazeSize**(1/3.0))
        # Generate maze.
        self.walls = mazeGenerator(self.rows, self.cols, self.lays)
        # Get maze data.
        self.cellDistances = cellDistancesFromSource(self.walls, self.rows,
                self.cols, self.lays, (0, 0, 0))
        self.cellColors = keyColorsFromScores(self.cellDistances)
        self.cellGrays = keyGraysFromScores(self.cellDistances)
        self.deadEnds = deadEnds(self.walls, self.rows, self.cols, self.lays)
        self.deadEnds.discard((0, 0, 0))

    def initVisibleGeometry(self):
        self.cellSize, self.thickness = 10, 0.2
        # Generate maze model.
        self.mazeColor = render.attachNewNode(GeomNode('maze gnode'))
        self.mazeGray = render.attachNewNode(GeomNode('maze gnode'))
        for wall in self.walls:
            corners = cubeCornersFromWall(wall, self.cellSize, self.thickness)
            # Add wall to geom node.
            cube = cubeFromCorners(corners[0], corners[1], self.cellColors,
                    self.cellSize)
            self.mazeColor.node().addGeom(cube)
            cube = cubeFromCorners(corners[0], corners[1], self.cellGrays,
                    self.cellSize)
            self.mazeGray.node().addGeom(cube)
        self.mazeGray.hide()
        self.initFlowers()
        # Enable per-pixel lighting, shaders, light ramp.
        render.setColorOff()
        render.setShaderAuto()
        render.setAttrib(LightRampAttrib.makeHdr1())

    def initFlowers(self):
        # Generate flowers.
        self.flowers = dict()
        fractalFunctions = [sierpinskiTetrahedron, kochSnowflake,
                sierpinskiDodecahedron, sierpinskiIcosahedron, cubeFlake,
                mengerSponge, vicsekFractal, pythagorasFlake]
        random.shuffle(fractalFunctions)
        fnIndex = 0
        for cell in self.deadEnds:
            self.flowers[cell] = Flower(cell)
            # Pick function.
            self.flowers[cell].fn = fractalFunctions[fnIndex]
            fnIndex = (fnIndex + 1) % len(fractalFunctions)
            # Init geometry.
            self.flowers[cell].np = self.flowers[cell].fn(0)
            self.flowers[cell].np.reparentTo(render)
            # Set color, position.
            self.flowers[cell].np.setColor(*self.cellColors[cell])
            self.flowers[cell].np.setPos(*getCellCenter(self.cellSize, *cell))
            # Add light.
            self.initFlowerLight(cell)

    def initFlowerLight(self, cell):
        self.flowers[cell].light = self.flowers[cell].np.attachNewNode(
                PointLight('self.flowerLight[%s]' % str(cell)))
        self.flowers[cell].light.node().setColor((0, 0, 0, 1))
        self.flowers[cell].light.node().setAttenuation((1,0,0.001))
        self.flowers[cell].light.setPos(0, 0, 0)
        # Tell everything to become enlightened by the player's light.
        render.setLight(self.flowers[cell].light)

    def initCameraLight(self):
        # Make the player glow.
        cameraLight = self.camera.attachNewNode(PointLight('camera light'))
        cameraLight.node().setColor((1, 1, 1, 1))
        cameraLight.node().setAttenuation((1,0,0.001))
        cameraLight.setPos(0, 0, 0)
        # Tell everything to become enlightened by the player's light.
        render.setLight(cameraLight)

    def initCollisionGeometry(self):
        # Create collision polygons for walls.
        self.wallygons = render.attachNewNode(CollisionNode('maze cnode'))
        for wall in self.walls:
            corners = cubeCornersFromWall(wall, self.cellSize, self.thickness)
            addCollisionPolygonsFromCorners(corners[0], corners[1],
                    self.wallygons.node())
        # Set collide mask of "wallygons".
        self.wallygons.node().setFromCollideMask(BitMask32.allOff())
        self.wallygons.node().setIntoCollideMask(BitMask32.bit(0))

    def initCollisionDetection(self):
        # Initialize the collision traverser.
        self.cTrav = CollisionTraverser()
        self.cTrav.setRespectPrevTransform(True)
        # Initialize the handler.
        self.handler = CollisionHandlerFluidPusher()
        # Add a collision sphere around the camera.
        self.cnodePath = self.camera.attachNewNode(CollisionNode('cam cnode'))
        self.cnodePath.node().addSolid(CollisionSphere(0, 0, 0, 1.5))
        # Set collide mask so that it will collide into walls.
        self.cnodePath.node().setFromCollideMask(BitMask32.bit(0))
        self.cnodePath.node().setIntoCollideMask(BitMask32.allOff())
        # Add the handler to the traverser.
        self.cTrav.addCollider(self.cnodePath, self.handler)
        # Add our collision node to the handler.
        self.handler.addCollider(self.cnodePath, self.camera, self.drive.node())

    def initAudioManager(self):
        self.audio3d = Audio3DManager.Audio3DManager(base.sfxManagerList[0],
                self.camera)

    def initStereo(self):
        # Make stereo display region and let it show the view from the camera.
        self.sdr = self.win.makeStereoDisplayRegion()
        self.sdr.setCamera(self.cam)
        # Set red-cyan anaglyph.
        leftFilter = ColorWriteAttrib.CRed
        rightFilter = (ColorWriteAttrib.CBlue | ColorWriteAttrib.CGreen)
        self.win.setRedBlueStereo(True, leftFilter, rightFilter)
        # Clear depth buffers
        self.sdr.setClearDepthActive(True)
        # Set to mono for now.
        self.sdr.setStereoChannel(Lens.SCMono)

    def initCameraControl(self):
        # Set camera position, field of view, heading, and pitch.
        self.camera.setPos(*getCellCenter(self.cellSize, 0, 0, 0))
        self.camLens.setFov(80)
        # Turn off the default camera driver.
        self.disableMouse()
        if self.mouselook:
            self.mousex, self.mousey = 0, 0
        # Initialize variables for camera control task.
        self.lastTime = 0
        self.lastPos = getCellCenter(self.cellSize, 0, 0, 0)
        # For debugging:
        self.lastCell = getCellFromPoint(self.cellSize, *self.camera.getPos())
        self.speed = 10
        self.controls = {"forward": 0, "backward": 0, "right": 0, "left": 0,
                         "up": 0, "down": 0, "look up": 0, "look down": 0,
                         "look right": 0, "look left": 0}
        # Start the camera control task:
        taskMgr.add(self.controlCamera, "camera-task")

    def initBloomTask(self):
        msg = "%d flowers to bloom." % len(Flower.unbloomed)
        self.flowerMsg = OnscreenText(text=msg, scale=0.09, fg=(1,1,1,1),
                shadow=(0,0,0,0.5), parent=base.a2dTopRight,
                pos=(-0.04, -0.12), align=TextNode.ARight, mayChange=True)
        taskMgr.add(self.toBloomOrNotToBloom, "bloom-task")

    def initControls(self):
        keyControl = {"mouse1"      : "forward",
                      "mouse3"      : "backward",
                      "q"           : "up",
                      "e"           : "down",
                      "w"           : "forward",
                      "a"           : "left",
                      "s"           : "backward",
                      "d"           : "right",
                      "arrow_left"  : "look left",
                      "arrow_right" : "look right",
                      "arrow_up"    : "look up",
                      "arrow_down"  : "look down",
                     } 
        for key in keyControl:
            self.accept(key, self.setControl, [keyControl[key], True])
            self.accept(key + "-up", self.setControl, [keyControl[key], False])
        # To toggle anaglyph 3D.
        self.accept("3", self.toggleStereo)
        # Other keys.
        self.accept("space", self.releaseGhost)
        self.accept("shift-space", self.moveCameraLikeGhost)
        self.accept("tab", self.teleportToRandomFlower)
        self.acceptOnce("enter", self.bloomAllFlowers)

    def initInst(self):
        # Print instructions.
        inst0 = "H to toggle instructions."
        inst1 = """\
Bloom the flowers in the dark corners of the garden.
To bloom a flower, move close to it.
If lost, press Space, and a will o' the wisp will guide you."""
        if self.mouselook:
            inst1 += """

Move the cursor or use arrow keys to look around.
Left and right click to move forward and backward."""
        self.inst0 = onscreenMessage(1, inst0)
        self.inst1 = onscreenMessage(3, inst1)
        # controls.png generated with http://www.keyboard-layout-editor.com/
        vertScale = 0.6 if self.mouselook else 0.75
        horizScale = vertScale*1206/720.0
        self.inst2 = OnscreenImage(image='assets/controls.png',
                pos=(horizScale, 0, vertScale), parent=base.a2dBottomLeft,
                scale=(horizScale, 1, vertScale))
        self.inst2.setTransparency(TransparencyAttrib.MAlpha)
        self.inst1.hide()
        self.inst2.hide()
        # Hold H for help.
        def toggleInst(*args):
            for x in args:
                x.show() if x.node().isOverallHidden() else x.hide()
        self.accept("h", toggleInst, [self.inst1, self.inst2])

    def setControl(self, action, value):
        self.controls[action] = value

    def toggleStereo(self):
        if self.sdr.getStereoChannel():
            self.sdr.setStereoChannel(Lens.SCMono)
            self.mazeGray.hide()
            self.mazeColor.show()
            for cell in self.flowers:
                self.flowers[cell].np.setColor(*self.cellColors[cell])
        else:
            self.sdr.setStereoChannel(Lens.SCStereo)
            self.mazeColor.hide()
            self.mazeGray.show()
            for cell in self.flowers:
                self.flowers[cell].np.setColor(*self.cellGrays[cell])

    def teleportToRandomFlower(self):
        targets = (list(Flower.unbloomed) if Flower.unbloomed
                   else list(self.flowers.keys()))
        cell = random.choice(targets)
        neighbor = neighbors(self.walls, self.rows, self.cols, self.lays, cell)[0]
        self.camera.setPos(*getCellCenter(self.cellSize, *neighbor))
        self.camera.lookAt(self.flowers[cell].np.getPos(), LVector3(0, -1, 0))
        self.camera.setHpr(self.camera.getHpr())

    def bloomAllFlowers(self):
        for cell in Flower.unbloomed:
            self.doBloom(cell)
        Flower.unbloomed.clear()

    def releaseGhost(self):
        source = getCellFromPoint(self.cellSize, *self.camera.getPos())
        targets = (list(Flower.unbloomed) if Flower.unbloomed
                   else list(self.flowers.keys()))
        path = pathToNearestTarget(self.walls, self.rows, self.cols, self.lays,
                source, targets)
        dest = path[-1]

        def Ghost():
            ghost = self.flowers[dest].fn(0)
            ghost.reparentTo(render)
            ghost.setColor(*self.cellColors[dest])
            ghost.setPos(*getCellCenter(self.cellSize, *source))
            ghost.setTransparency(TransparencyAttrib.MAlpha)
            return ghost

        def GhostLight():
            # Ghost light.
            light = ghost.attachNewNode(PointLight('ghost light'))
            light.node().setAttenuation((1,0,0.001))
            light.setPos(0, 0, 0)
            render.setLight(light)
            return light

        def GhostFlicker():
            # Ghost flicker.
            def setBrightness(t):
                light.node().setColor((t, t, t, 1))
                maxT, minA, maxA = 0.2, 0.2, 0.8
                alpha = minA + (t/maxT)*(maxA-minA)
                ghost.setAlphaScale(alpha)
            flicker = Sequence(
                    LerpFunctionInterval(setBrightness, 1.5, 0, 0.2),
                    LerpFunctionInterval(setBrightness, 1.5, 0.2, 0))
            flicker.loop()
            return flicker

        def startGhostRotation():
            ghost.hprInterval(6, (360, 360*2, 360*3)).loop()

        def ghostStep(cell):
            move = ghost.posInterval(1.0, getCellCenter(self.cellSize, *cell))
            # Pause if turning.
            i = path.index(cell)
            if 2 <= i:
                difference = [path[i-2][j] - path[i][j] for j in range(3)]
                if difference.count(0) < 2:
                    return Sequence(Wait(2), move)
            return move

        ghost = Ghost()
        light = GhostLight()
        flicker = GhostFlicker()
        startGhostRotation()

        # Do ghost movement.
        Sequence(
            Sequence(*list(map(ghostStep, path))),
            Func(lambda: flicker.finish()),
            Func(lambda: light.removeNode()),
            Func(lambda: ghost.removeNode())
            ).start()

    def moveCameraLikeGhost(self):
        source = getCellFromPoint(self.cellSize, *self.camera.getPos())
        targets = (list(Flower.unbloomed) if Flower.unbloomed
                   else list(self.flowers.keys()))
        path = pathToNearestTarget(self.walls, self.rows, self.cols, self.lays,
                source, targets)
        def cameraStep(cell):
            move = self.camera.posInterval(1.0, getCellCenter(self.cellSize, *cell))
            # Pause if turning.
            i = path.index(cell)
            if 1 < i:
                difference = [path[i-2][j] - path[i][j] for j in range(3)]
                if difference.count(0) < 2:
                    difference = tuple([path[i][j] - path[i-1][j] for j in range(3)])
                    newHPR = False
                    if difference == (1, 0, 0):
                        newHPR = (0, 0, 0)
                    elif difference == (-1, 0, 0):
                        newHPR = (180, 0, 0)
                    elif difference == (0, 1, 0):
                        newHPR = (270, 0, 0)
                    elif difference == (0, -1, 0):
                        newHPR = (90, 0, 0)
                    if not newHPR:
                        return Sequence(Wait(2), move)
                    else:
                        return Sequence(self.camera.hprInterval(2, newHPR), move)
            return move
        Sequence(*list(map(cameraStep, path[:-1]))).start()

    def lookAround(self, task):
        heading, pitch, roll = tuple(self.camera.getHpr())
        # Looking around.
        if self.mouselook:
            # Find the mouse delta in pixels.
            pointer = self.win.getPointer(0)
            dx, dy = pointer.getX(), pointer.getY()
            # Set new heading and pitch.
            if self.win.movePointer(0, 100, 100):
                visionScalar = 0.2
                if not self.inverted:
                    heading = heading - (dx - 100) * visionScalar
                else:
                    heading = heading + (dx - 100) * visionScalar
                pitch = pitch - (dy - 100) * visionScalar
        lookStep = 2
        if ((self.controls["look left"] and not self.inverted)
                or (self.controls["look right"] and self.inverted)):
            heading = heading + lookStep
        if ((self.controls["look right"] and not self.inverted)
                or (self.controls["look left"] and self.inverted)):
            heading = heading - lookStep
        if self.controls["look up"]:
            pitch = pitch + lookStep
        if self.controls["look down"]:
            pitch = pitch - lookStep
        # Make sure you can't invert twice.
        if pitch <= -270: pitch = +90
        if pitch >= 270: pitch = -90
        # Actually set the orientation.
        self.camera.setHpr(heading, pitch, 0)
        
    def moveAround(self, task):
        orientation = self.camera.getMat().getRow3(1)
        direction = LVecBase3f(0, 0, 0)
        elapsed = 0 if (self.lastTime == 0) else (task.time - self.lastTime)
        if self.controls["forward"]:
            direction = orientation
        if self.controls["backward"]:
            direction = -orientation
        if ((self.controls["up"] and not self.inverted)
                or (self.controls["down"] and self.inverted)):
            direction = LVecBase3f(0, 0, 1)
        if ((self.controls["down"] and not self.inverted)
                or (self.controls["up"] and self.inverted)):
            direction = LVecBase3f(0, 0, -1)
        if ((self.controls["right"] and not self.inverted)
                or (self.controls["left"] and self.inverted)):
            direction = LVecBase3f(orientation.getY(), -orientation.getX(), 0)
        if ((self.controls["left"] and not self.inverted)
                or (self.controls["right"] and self.inverted)):
            direction = LVecBase3f(-orientation.getY(), orientation.getX(), 0)
        if self.controls["right"] or self.controls["left"]:
            # Make sure direction is a unit vector.
            direction /= direction.length()
        self.camera.setFluidPos(self.camera.getPos() +
                direction * elapsed * self.speed)
    
    def flowerForceField(self, task):
        x, y, z = tuple(self.camera.getPos())
        for cell in self.flowers:
            x1, y1, z1 = getCellCenter(self.cellSize, *cell)
            distance = math.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2)
            if distance < self.cellSize//2 - 1.5:
                self.camera.setPos(self.lastPos)

    def controlCamera(self, task):
        cell = getCellFromPoint(self.cellSize, *self.camera.getPos())
        if cell != self.lastCell:
            print(cell, self.cellDistances[cell])
        self.lastCell = cell

        heading, pitch, roll = tuple(self.camera.getHpr())
        self.inverted = not(-90 <= pitch <= 90)
        self.lookAround(task)
        self.moveAround(task)
        self.flowerForceField(task)

        self.lastTime = task.time
        self.lastPos = self.camera.getPos()
        return Task.cont

    def toBloomOrNotToBloom(self, task):
        cell = getCellFromPoint(self.cellSize, *self.camera.getPos())
        if cell in Flower.unbloomed:
            self.doBloom(cell)
            Flower.unbloomed.discard(cell)
        self.flowerMsg.setText(
                "%d flowers to bloom." % len(Flower.unbloomed)
                if Flower.unbloomed else "All flowers have bloomed!")
        return Task.cont

    def doSound(self, source, path):
        sound = self.audio3d.loadSfx(path)
        # Enable doppler shifting.
        self.audio3d.setSoundVelocityAuto(sound)
        self.audio3d.setListenerVelocityAuto()
        # Attach sound to flower and play it.
        self.audio3d.attachSoundToObject(sound, source)
        sound.play()

    def doBloom(self, cell):

        def startRotation():
            h, p, r = tuple(self.flowers[cell].np.getHpr())
            self.flowers[cell].spin = self.flowers[cell].np.hprInterval(60,
                    (h+360, p+360*2, r+360*3))
            self.flowers[cell].spin.loop()

        def changeLevel(level=None):
            color = self.flowers[cell].np.getColor()
            pos = self.flowers[cell].np.getPos()
            hpr = self.flowers[cell].np.getHpr()
            self.flowers[cell].np.removeNode()
            if level != None:
                self.flowers[cell].np = self.flowers[cell].fn(level)
            else:
                self.flowers[cell].np = self.flowers[cell].fn()
            self.flowers[cell].np.reparentTo(render)
            self.flowers[cell].np.setColor(color)
            self.flowers[cell].np.setPosHpr(pos, hpr)
            startRotation()

        def doGrow():
            from functools import reduce
            Sequence(*reduce(lambda x, y: x + [Wait(1), Func(changeLevel, y)],
                list(range(1, 6)), [])).start()

        def startFlicker():
            # Make light flicker once every three seconds.
            setFlowerBrightness = (lambda t:
                    self.flowers[cell].light.node().setColor((t, t, t, 1)))
            self.flowers[cell].flicker = Sequence(
                    LerpFunctionInterval(setFlowerBrightness, 1.5, 0, 0.2),
                    LerpFunctionInterval(setFlowerBrightness, 1.5, 0.2, 0))
            self.flowers[cell].flicker.loop()

        self.doSound(self.flowers[cell].light, 'assets/sparkle.wav')
        # Sound from https://www.freesound.org/people/YleArkisto/sounds/369610/
        doGrow()
        startFlicker()
        startRotation()

from tkinter import *
import tkinter.simpledialog, tkinter.messagebox

class SettingsDialog(tkinter.simpledialog.Dialog):
    # Code adapted from:
    # http://effbot.org/tkinterbook/tkinter-dialog-windows.htm
    def ok(self, event=None):
        if not self.validate():
            self.initial_focus.focus_set() # put focus back
            return
        self.apply()
        self.withdraw()
        self.update_idletasks()
        self.destroy()
    def cancel(self, event=None):
        self.result = None
        self.destroy()
    def body(self, master):
        self.title("Maze Settings")
        Label(master, text="Rows").grid(row=0, column=0)
        Label(master, text="Columns").grid(row=0,column=1)
        Label(master, text="Layers").grid(row=0,column=2)
        self.e1, self.e2, self.e3 = Entry(master), Entry(master), Entry(master)
        self.e1.grid(row=1, column=0)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=1, column=2)
        Label(master,
                text="Leave any of the above blank to use default dimensions."
                ).grid(row=2, columnspan=3)
        self.var = IntVar()
        self.cb = Checkbutton(master, text="Mouselook", variable=self.var)
        self.cb.grid(row=4, columnspan=3)
    def validate(self):
        mouse = self.var.get()
        try:
            rows = int(self.e1.get())
            cols = int(self.e2.get())
            lays = int(self.e3.get())
        except:
            self.result = ("Default", mouse)
            tkinter.messagebox.showinfo("Message", "Proceeding with default dimensions.")
            return True
        if (rows*cols*lays) <= 1:
            tkinter.messagebox.showinfo("Message", "Maze is too small.")
            return False
        elif (rows*cols*lays) >= sys.getrecursionlimit():
            tkinter.messagebox.showinfo("Message", "Maze is too large.")
            return False
        else:
            self.result = (rows, cols, lays, mouse)
            return True

root = Tk()
root.withdraw()
conf = SettingsDialog(root).result
root.destroy()

if conf == None:
    sys.exit()
if conf[0] == "Default":
    MazeExplorer(mouselook=conf[1]).run()
else:
    MazeExplorer(*conf).run()
