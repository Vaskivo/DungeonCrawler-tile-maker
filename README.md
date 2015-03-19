DungeonCrawler-tile-maker
=========================

Generates wall tiles to be used in a 'old school dungeon crawler' like Bard's Tale, Eye of the Beholder and Might And Magic (1-5). 


How to use
----------

You must have an image with a 'front wall'. Feed it to the function and it will generate the tiles.

```python
generate_tiles(source_wall_filename, 			# Filename of the original image
			   result_name, 					# Folder where the tiles will be generated to
			   wall_dims, 						# dimensions of the wall - tuple(width, height)
			   sides, 							# how many cells will be considered and calculated to 
												# the left and to the right. (it will correspond to how
												# many cells are available to be rendered)
			   depth, 							# how many cells will be considered and calculated 
												# 'forward'. (it will correspond to how
												# many cells are available to be rendered)
			   depth_offset, 					# The camera is centerend inside the cell at (0,0,0). 
												# This value displaces all the cells 'forward'
			   screen_dims, 					# dimensions of the screen - tuple(width, height)
			   horizontal_fov=90, 				# horizontal field of view angle, in degrees
			   vertical_fov=60					# vertical field of view angle, in degrees
			   )
			  
```

Just run the file. It has an example in it.


Show me stuff!
--------------

From this:
![Original Image](wall.png "From this")

To this:
![Composition of the resulting images](result.png "To this.")

In the pic above I composed the generated images to show the 'illusion of depth' achieved.


Dependencies
------------

- I started this in Python 2.7 but finished it in 3.3 (don't ask). Still, I believe it runs in both.
- Pillow
- Numpy


Credits
-------

The source image in the repo can be found [here](http://opengameart.org/node/10606) (thanks xmorg), although I butchered it a bit to suit my tests.


TODO
----

- Better documentation
- Cropping images.
- Add padding to make the size of the images a power of two

