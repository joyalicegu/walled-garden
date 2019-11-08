This is a game.

Bloom the flowers in the dark corners of the walled garden.
To bloom a flower, move close to it.
If lost, press Space, and a will o' the wisp will guide you.

See the demo video [here](https://www.youtube.com/watch?v=wHIEGukRrDE).

---

This was originally written in 2016 for a class I was taking.
It was written in Python 2 (deprecated) and used Panda3D v1.9.2 (obsolete).

In November 2019, it was lazily ported to Python 3 with 2to3.
It seems to work fine with Panda3D v1.10.4.1.

---

Things I don't like:

- The camera and movement controls respect an up-down axis that doesn't change
  based on the player's perspective, so they behave unintuitively in some
  camera orientations, in a way that can be jarring for the player.
- The colors on the walls don't blend together.
- The polygons making up the walls overlap.
- It's slow, especially after more fractals have been expanded.

Things I still like:

- The fractals, 3D anaglyph mode, and 3D mazes are still cool.
- Being able to press Shift-Space to auto-follow the so-called "will o' the
  wisp" is cool. I originally added it as a secret thing for demos or
  debugging, but it's pretty nice to just sit back and relax as the camera
  moves through the maze.
