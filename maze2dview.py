from mazegen import mazeGenerator
from Tkinter import *

def drawWall(wall, canvas, cellSize, margin):
    rows = [cell[0] for cell in wall]
    cols = [cell[1] for cell in wall]
    if rows[0] == rows[1]:
        x0 = max(cols)
        y0 = rows[0]
        x1 = max(cols)
        y1 = rows[0] + 1
    else: 
        x0 = cols[0]
        y0 = max(rows)
        x1 = cols[0] + 1
        y1 = max(rows)
    splat = tuple(map(lambda x: cellSize*x + margin, [x0, y0, x1, y1]))
    canvas.create_line(*splat)

import sys, collections, random

maxRowOrCol = int(sys.getrecursionlimit()**0.5)
rows = min(30, maxRowOrCol)
cols = min(30, maxRowOrCol)

cellSize, margin = 20, 10

root = Tk()
canvas = Canvas(root, width=cols*cellSize+2*margin, height=rows*cellSize+2*margin)
canvas.pack()
walls = mazeGenerator(rows, cols)

def cellDistancesFromSource(walls, source):
    currentCells = collections.deque()
    currentCells.append(source)
    visited = set()
    visited.add(source)
    dist = dict()
    dist[source] = 0
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    while len(currentCells) != 0:
        currentCell = currentCells.popleft()
        (row, col) = currentCell
        nextCells = list()
        for (drow, dcol) in directions:
            nextCell = (row + drow, col + dcol)
            if {nextCell, currentCell} not in walls:
                nextCells.append(nextCell)
        for nextCell in nextCells:
            if nextCell not in visited:
                dist[nextCell] = dist[currentCell] + 1
                visited.add(nextCell)
                currentCells.append(nextCell)
    return dist

def hsvToRgb(h, s, v):
    # Formula from https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
    x = (1 - abs(h/60.0 % 2 - 1))
    r, g, b = {0: (1, x, 0), # red to yellow (more red than green)
               1: (x, 1, 0), # yellow to green (more green than red)
               2: (0, 1, x), # green to cyan (more green than blue)
               3: (0, x, 1), # cyan to blue (more blue than green)
               4: (x, 0, 1), # blue to purple (more blue than red)
               5: (1, 0, x)  # purple to red (more red than blue)
               }[h/60] 
    return tuple(map(lambda x: v*(x*s+1-s), [r, g, b]))

def keyColorsFromScores(scores):
    minScore, maxScore = min(scores.values()), max(scores.values())
    minHue, maxHue = 0, 300 # red to purple
    step = (maxHue-minHue) / float(maxScore-minScore)
    colors = dict()
    for key in scores:
        hue = int(minHue + (scores[key] - minScore) * step)
        (r, g, b) = hsvToRgb(hue, 1.0, 1.0)
        colors[key] = "#%02x%02x%02x" % (int(r*255),int(g*255),int(b*255))
    return colors

def pathToNearestTarget(walls, rows, cols, source, targets):
   targets = set(targets)
   currentCells = collections.deque()
   currentCells.append(source)
   visited = set()
   visited.add(source)
   path = dict()
   path[source] = [source]
   directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
   while len(currentCells) != 0:
       currentCell = currentCells.popleft()
       (row, col) = currentCell
       nextCells = list()
       for (drow, dcol) in directions:
           nextCell = (row + drow, col + dcol)
           if {nextCell, currentCell} not in walls:
               nextCells.append(nextCell)
       for nextCell in nextCells:
           if nextCell not in visited:
               visited.add(nextCell)
               currentCells.append(nextCell)
               path[nextCell] = path[currentCell] + [nextCell]
               if nextCell in targets:
                   return path[nextCell]

colors = keyColorsFromScores(cellDistancesFromSource(walls, (0, 0)))

for row in range(rows):
    for col in range(cols):
        x0 = margin + cellSize * col
        y0 = margin + cellSize * row
        x1 = margin + cellSize * (col + 1)
        y1 = margin + cellSize * (row + 1)
        canvas.create_rectangle(x0, y0, x1, y1, fill=colors[(row, col)], width=0)

for wall in walls:
    drawWall(wall, canvas, cellSize, margin)

targets = random.sample([(row, col) for row in range(rows) for col in range(cols)], 5)
path = pathToNearestTarget(walls, rows, cols, (0,0), targets)

for cell in path:
    row, col = cell
    x0 = margin + cellSize * col + 5
    y0 = margin + cellSize * row + 5
    x1 = margin + cellSize * (col + 1) - 5
    y1 = margin + cellSize * (row + 1) - 5
    canvas.create_oval(x0, y0, x1, y1, fill="white", width=0)

for cell in targets:
    row, col = cell
    x0 = margin + cellSize * col + 5
    y0 = margin + cellSize * row + 5
    x1 = margin + cellSize * (col + 1) - 5
    y1 = margin + cellSize * (row + 1) - 5
    canvas.create_oval(x0, y0, x1, y1, fill="black", width=0)

root.mainloop()
