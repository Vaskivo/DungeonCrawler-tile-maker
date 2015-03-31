

from __future__ import division
from collections import namedtuple, defaultdict
from PIL import Image, ImageDraw
import numpy
import math
import json
import os

Point3D = namedtuple('Point3D', 'x y z')
Point2D = namedtuple('Point2D', 'x y')

Trapeze = namedtuple('Trapeze', 'tl tr br bl')

# the wall location is in "cell space" (i.e. (-1,0) means it's 
# the cell at the left of the camera.
# Coordinates are (y, x), where y is forward and x is to the right
Wall = namedtuple('Wall', 'corners location side')

def _recursive_default_dict():
    f = lambda: defaultdict(f)
    return defaultdict(f)
    

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
    

def _get_wall_name(wall):
    locs = map(lambda x: str(x), wall.location)
    return '{0}_{1}'.format( '_'.join(locs), wall.side)
    
    

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
    
    walls = []
    for row in range(depth):
        for column in range(1-sides, sides):
            point = (row, column)
            walls.append( Wall(world_wall_coordinates(eqs['f'], row, column, u, v, depth_offset),
                               point,
                               'f'
                              ))
            if column <= 0:     #left wall
                walls.append( Wall(world_wall_coordinates(eqs['l'], row, column, u, v, depth_offset),
                                   point,
                                   'l'
                                  ))           
            if column >= 0:     # right wall
                walls.append( Wall(world_wall_coordinates(eqs['r'], row, column, u, v, depth_offset),
                                   point,
                                   'r'
                                  ))
                
    return walls
    

def world_walls_in_screen(world_walls, screen_dims, horizontal_fov, vertical_fov):
    # calculations from http://www.extentofthejam.com/pseudo/

    horizontal_fov = math.radians(horizontal_fov)
    vertical_fov = math.radians(vertical_fov)
    
    horizontal_scaling = screen_dims[0] / math.tan(horizontal_fov/2)
    vertical_scaling = screen_dims[1] / math.tan(vertical_fov/2)
    
    screen_walls = []
    for wall in world_walls:
        new_polygon = []
        for point_3d in wall.corners:
            point_2d = Point2D( (point_3d.x * horizontal_scaling) / point_3d.z,
                                (point_3d.y * vertical_scaling) / point_3d.z)
            new_polygon.append(point_2d)
        screen_walls.append( Wall(Trapeze(*new_polygon), wall.location, wall.side))
        
    return screen_walls
       
     
def screen_culled_walls(screen_walls, screen_dims):
    w, h = screen_dims
    left, right = -w/2, w/2
    top, bottom = h/2, -h/2
        
    new_walls = []
    for wall in screen_walls:
        corners = wall.corners
        if min(corners.tl.x, corners.bl.x) > right or \
           max(corners.tr.x, corners.br.x) < left or \
           max(corners.tl.y, corners.tr.y) < bottom or \
           min(corners.bl.y, corners.br.y) > top:
            continue
        new_walls.append(wall)
        
    return new_walls
    
     
def generate_wall_images(wall_filename, result_folder, screen_walls, screen_dims, crop=False):
    screen_w, screen_h = screen_dims
    half_screen_w = screen_w/2
    half_screen_h = screen_h/2

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
    
    for wall in screen_walls:
        wall_img_coords = []
        for corner in wall.corners:
            wall_img_coords.append(Point2D(corner.x + half_screen_w, half_screen_h - corner.y))
        wall_img_coords = Trapeze(*wall_img_coords)
        
        coeffs = _find_coeffs(wall_img_coords, source_image_coords)
        new_image = source_image.transform(screen_dims, 
                                           Image.PERSPECTIVE,
                                           coeffs,
                                           Image.BICUBIC)
                                           
        if crop:
            left = min( (p.x for p in wall_img_coords) )
            top = min( (p.y for p in wall_img_coords) )
            right = max( (p.x for p in wall_img_coords) )
            bottom = max( (p.y for p in wall_img_coords) )

            left = max(left, 0)
            top = max(top, 0)
            right = min(right, screen_w)
            bottom = min(bottom, screen_h)
            
            box = (left, top, right, bottom)
            box = tuple(int(round(x)) for x in box)

            new_image = new_image.crop(box)
            
        new_image.save(os.path.join(result_folder, _get_wall_name(wall)) + '.png', 'PNG')
    

def generate_json_file(result_name, screen_dims, screen_walls, crop=False):
    data = _recursive_default_dict()
    data['image_size'] = [screen_dims[0], screen_dims[1]]
    
    tiles = data['tiles']
    for wall in screen_walls:
        tile = tiles[_get_wall_name(wall)]
        
        wall_corners = wall.corners
        
        # location of the wall in "cell space"
        tile['location'] = { 'y' : wall.location[0],
                             'x' : wall.location[1]
                           }
        tile['side'] = wall.side
        
        # this reflects data of the wall in the "screen space"
        corners = tile['corners']
        corners['top_left'] = list(wall_corners.tl)
        corners['top_right'] = list(wall_corners.tr)
        corners['bottom_right'] = list(wall_corners.br)
        corners['bottom_left'] = list(wall_corners.bl)
        
        sides = tile['sides']
        sides['left'] = min([point.x for point in wall_corners])
        sides['right'] = max([point.x for point in wall_corners])
        sides['top'] = max([point.y for point in wall_corners])
        sides['bottom'] = min([point.y for point in wall_corners])
        
        tile['center'] = [ (sides['left'] + sides['right']) / 2,
                           (sides['top'] + sides['bottom']) / 2]
                           
        tile['size'] = [ (sides['right'] - sides['left']),
                         (sides['top'] - sides['bottom']) ]
                         
        if crop:
            # this reflects data about the cropped image (size, where it should go to, etc)
            screen_left = -screen_dims[0]/2
            screen_top = screen_dims[1]/2
            screen_right = screen_dims[0]/2
            screen_bottom = -screen_dims[1]/2
            
            img_data = tile['image_data']
            img_sides = img_data['sides']
            img_sides['left'] = max(sides['left'], screen_left)
            img_sides['top'] = min(sides['top'], screen_top)
            img_sides['right'] = min(sides['right'], screen_right)
            img_sides['bottom'] = max(sides['bottom'], screen_bottom)
            
            img_data['center'] = [ (img_sides['left'] + img_sides['right']) / 2,
                                   (img_sides['top'] + img_sides['bottom']) / 2 ]
            
            img_data['size'] = [ (img_sides['right'] - img_sides['left']),
                                 (img_sides['top'] - img_sides['bottom']) ]
            
            
    
    filename = os.path.join(result_name, 'data.json')
    with open(filename, 'w') as output:
        json.dump(data, output, sort_keys=True, indent=4)
        
       
       
       
def generate_tiles(source_wall_filename, result_name, wall_dims, sides, 
                    depth, depth_offset, screen_dims, crop=False, horizontal_fov=90, vertical_fov=60):
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
    w = screen_culled_walls(w, screen_dims)
    generate_wall_images(source_wall_filename, result_name, w, screen_dims, crop)
    generate_json_file(result_name, screen_dims, w, crop)
      
       
        
if __name__ == '__main__':
    
    generate_tiles('wall.png', 'coisas', (50, 40), 3, 4, 50, 
                    (650, 480), crop=False, horizontal_fov=90, vertical_fov=60)
    
        
    
    
            
            
            
            
            
            
            
            
            
            