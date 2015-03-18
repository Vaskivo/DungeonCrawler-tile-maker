

from __future__ import division
from collections import namedtuple
from PIL import Image, ImageDraw
import numpy
import math
import os

Point3D = namedtuple('Point3D', 'x y z')
Point2D = namedtuple('Point2D', 'x y')

Trapeze = namedtuple('Trapeze', 'tl tr br bl')


eqs = {
    'f' : { 'tl': lambda row, column, u, v, z_offset: Point3D( (-u/2) + column * u,
                                                               (v/2),
                                                               (u/2) + row * u + z_offset
                                                             ),
            'tr': lambda row, column, u, v, z_offset: Point3D( (u/2) + column * u,
                                                               (v/2),
                                                               (u/2) + row * u + z_offset
                                                             ),
            'bl': lambda row, column, u, v, z_offset: Point3D( (-u/2) + column * u,
                                                               (-v/2),
                                                               (u/2) + row * u + z_offset
                                                             ),
            'br': lambda row, column, u, v, z_offset: Point3D( (u/2) + column * u,
                                                               (-v/2),
                                                               (u/2) + row * u + z_offset
                                                             )
          },
    'l' : { 'tl': lambda row, column, u, v, z_offset: Point3D( (-u/2) + column * u,
                                                               (v/2),
                                                               (-u/2) + row * u + z_offset
                                                             ),
            'tr': lambda row, column, u, v, z_offset: Point3D( (-u/2) + column * u,
                                                               (v/2),
                                                               (u/2) + row * u + z_offset
                                                             ),
            'bl': lambda row, column, u, v, z_offset: Point3D( (-u/2) + column * u,
                                                               (-v/2),
                                                               (-u/2) + row * u + z_offset
                                                             ),
            'br': lambda row, column, u, v, z_offset: Point3D( (-u/2) + column * u,
                                                               (-v/2),
                                                               (u/2) + row * u + z_offset
                                                             )
          },
    'r' : { 'tl': lambda row, column, u, v, z_offset: Point3D( (u/2) + column * u,
                                                               (v/2),
                                                               (u/2) + row * u + z_offset
                                                             ),
            'tr': lambda row, column, u, v, z_offset: Point3D( (u/2) + column * u,
                                                               (v/2),
                                                               (-u/2) + row * u + z_offset
                                                             ),
            'bl': lambda row, column, u, v, z_offset: Point3D( (u/2) + column * u,
                                                               (-v/2),
                                                               (u/2) + row * u + z_offset
                                                             ),
            'br': lambda row, column, u, v, z_offset: Point3D( (u/2) + column * u,
                                                               (-v/2),
                                                               (-u/2) + row * u + z_offset
                                                             )
          },
    }
                         
                         
def world_wall_coordinates(funcs, row, column, u, v, z_offset=0):
    pos = {}
    for name, func in funcs.items():
        pos[name] = func(row, column, u, v, z_offset) 
    return Trapeze(pos['tl'], pos['tr'], pos['br'], pos['bl'])
      
      
def world_walls(face_dims, sides, depth, depth_offset=0):
    u, v = face_dims
    
    walls = {}
    for row in range(depth):
        for column in range(1-sides, sides):
            front_name = '{0}_{1}_f'.format(row, column)
            walls[front_name] = world_wall_coordinates(eqs['f'], row, column, u, v, depth_offset)
            
            if column <= 0:     #left wall
                name = '{0}_{1}_l'.format(row, column)
                walls[name] = world_wall_coordinates(eqs['l'], row, column, u, v, depth_offset)   
            
            if column >= 0:     # right wall
                name = '{0}_{1}_r'.format(row, column)
                walls[name] = world_wall_coordinates(eqs['r'], row, column, u, v, depth_offset)
                
    return walls
    

def world_walls_in_screen(world_walls, screen_dims, horizontal_fov, vertical_fov):
    # calculations from http://www.extentofthejam.com/pseudo/

    horizontal_fov = math.radians(horizontal_fov)
    vertical_fov = math.radians(vertical_fov)
    
    horizontal_scaling = screen_dims[0] / math.tan(horizontal_fov/2)
    vertical_scaling = screen_dims[1] / math.tan(vertical_fov/2)
    
    screen_walls = {}
    for name, wall in world_walls.items():
        new_polygon = []
        for point_3d in wall:
            point_2d = Point2D( (point_3d.x * horizontal_scaling) / point_3d.z,
                                (point_3d.y * vertical_scaling) / point_3d.z)
            new_polygon.append(point_2d)
        screen_walls[name] = Trapeze(*new_polygon)
    return screen_walls
       
     
def _find_coeffs(pa, pb):
    ''' Finds coefficients to be used by Image.transform '''
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

     
def generate_wall_images(wall_filename, result_folder, screen_walls, screen_dims):

    half_screen_w = screen_dims[0]/2
    half_screen_h = screen_dims[1]/2

    source_image = Image.open(wall_filename)
    image_w, image_h = source_image.size
    half_image_w = image_w/2
    half_image_h = image_h/2
    
    source_image_coords = Trapeze(Point2D(0, 0),
                                  Point2D(image_w, 0),
                                  Point2D(image_w, image_h),
                                  Point2D(0, image_h)
                                 )
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    for name, wall in screen_walls.items():
        wall_img_coords = []
        for point in wall:
            wall_img_coords.append(Point2D(point.x + half_screen_w, half_image_h - point.y))
        wall_img_coords = Trapeze(*wall_img_coords)
        
        coeffs = _find_coeffs(wall_img_coords, source_image_coords)
        new_image = source_image.transform(screen_dims, 
                                           Image.PERSPECTIVE,
                                           coeffs,
                                           Image.BICUBIC)

        new_image.save(os.path.join(result_folder, name) + '.png', 'PNG')
       
       
def generate_tiles(source_wall_filename, result_name, wall_dims, sides, 
                    depth, depth_offset, screen_dims, horizontal_fov=90, vertical_fov=60):
    """ Creates a folder with all the tiles.
    
    This functions creates a (kind of) 3D representation of the walls. With 
    this representation it then projects them into the 'screen plane'. It 
    uses the walls' screen coordinates to generate the tiles. The generated
    tiles are then outputted to a folder names 'result_name'.
    
    Args:
        source_wall_filename: file name of the image used for the walls
        result_name: name of the folder where all the tiles will be outputted
        wall_dims: dimensions of the wall - tuple(width, height)
        sides: how many cells will be considered and calculated to the left and
            to the right, i.e., the width of our '3D representation'
        depth: how many cells will be considered 'forward', i.e., the depth
            or our '3D representation'
        depth_offset: The camera is centerend inside the cell at (0,0,0). This
            value displaces all the cells 'forward', in world units
        screen_dims: dimensions of the screen - tuple(width, height)
        horizontal_fov: horizontal field of view angle, in degrees
        vertical_fov: vertical field of view angle, in degrees
    """            
                    
    w = world_walls(wall_dims, sides, depth, depth_offset)
    w = world_walls_in_screen(w, screen_dims, horizontal_fov, vertical_fov)
    generate_wall_images(source_wall_filename, result_name, w, screen_dims)
      
       
        
if __name__ == '__main__':
    
    generate_tiles('wall.png', 'coisas', (50, 40), 3, 3, 50, 
                    (300, 200), horizontal_fov=90, vertical_fov=60)
    
        
    
    
            
            
            
            
            
            
            
            
            
            