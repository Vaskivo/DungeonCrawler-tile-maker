

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
                         
                         
def get_world_wall_coordinates(funcs, row, column, u, v, z_offset=0):
    pos = {}
    for name, func in funcs.items():
        pos[name] = func(row, column, u, v, z_offset) 
    return Trapeze(pos['tl'], pos['tr'], pos['br'], pos['bl'])
      
      
def get_world_walls(face_dims, sides, depth, depth_offset=0):
    u, v = face_dims
    
    walls = {}
    for row in range(depth):
        for column in range(1-sides, sides):
            front_name = '{0}_{1}_f'.format(row, column)
            walls[front_name] = get_world_wall_coordinates(eqs['f'], row, column, u, v, depth_offset)
            
            if column <= 0:     #left wall
                name = '{0}_{1}_l'.format(row, column)
                walls[name] = get_world_wall_coordinates(eqs['l'], row, column, u, v, depth_offset)   
            
            if column >= 0:     # right wall
                name = '{0}_{1}_r'.format(row, column)
                walls[name] = get_world_wall_coordinates(eqs['r'], row, column, u, v, depth_offset)
                
    return walls


'''
y_screen = (y_world*scaling)/z_world + (y_resolution/2)

x_screen=640   fov_angle=60   y_world=sin(60/2)   z_world=(60/2)   x_resolution/2=320   scaling=?

x_screen = (y_world*scaling)/z_world + (x_resolution/2)
640 = (sin(30)*scaling/cos(30)) + 320
320 = tan(30)*scaling
320/tan(30) = scaling

In generic terms: scaling = (x_resolution/2) / tan(fov_angle/2) 


y_screen = (y_world*scaling)/z_world + (y_resolution/2)
x_screen = (y_world*scaling)/z_world + (x_resolution/2)
'''


    
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
                    

def calculate_screen_coordinates(screen_dims, point_3d, horizontal_angle, vertical_angle):
    horizontal_fov = math.radians(horizontal_angle)
    vertical_fov = math.radians(vertical_angle)
    
    horizontal_scaling = screen_dims[0] / math.tan(horizontal_fov/2)
    vertical_scaling = screen_dims[1] / math.tan(vertical_fov/2)
    
    return Point2D( (point_3d.x * horizontal_scaling) / point_3d.z,
                    (point_3d.y * vertical_scaling) / point_3d.z)
    
    
def get_polygon_screen_coordinates(screen_dims, polygon, horizontal_angle, vertical_angle): # polygon is a trapeze or tuple
    new_polygon = []
    for value in polygon:  
        new_polygon.append(calculate_screen_coordinates(screen_dims, value, horizontal_angle, vertical_angle))
    return Trapeze(*new_polygon)
    
    
def polygons_to_screen_coordinates(world_walls, screen_dims, horizontal_angle, vertical_angle):
    screen_walls = {}
    for name, wall in world_walls.items():
        screen_walls[name] = get_polygon_screen_coordinates(screen_dims, wall, horizontal_angle, vertical_angle)
        
    return screen_walls
       
       
def generate_wall_images(wall_filename, screen_walls, screen_dims):

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
    
    result_folder = 'coisas'
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
       
       
       
        
if __name__ == '__main__':
    
    w = get_world_walls((50, 40), 3, 3, 50)
    
    w = polygons_to_screen_coordinates(w, (300, 200), 90, 60)
    
    generate_wall_images('wall.png', w, (300, 200))
    
        
    
    
            
            
            
            
            
            
            
            
            
            