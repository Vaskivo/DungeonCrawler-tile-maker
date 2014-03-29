""" 
    TILEGEN 

    - This scripts generates tiles to be used in a first person tile based
      dungeon crawler games (like Eye of the Beholder and Might and Magic)

    - With a single image file, it generates all the ones needed

"""

from __future__ import division
from collections import namedtuple
from itertools import count
from PIL import Image
import os
import json
import numpy

Point = namedtuple('Point', 'x y')
Trapeze = namedtuple('Trapeze', 'tl tr br bl')

def trapeze_size(trapeze):
    return (max(trapeze.tr.x, trapeze.br.x) - min(trapeze.tl.x, trapeze.bl.x),
            max(trapeze.bl.y, trapeze.br.y) - min(trapeze.tl.y, trapeze.tr.y))


def trapeze_to_dict(trapeze):
    return dict(( ('top_left', tuple(trapeze.tl)),
                  ('top_right', tuple(trapeze.tr)),
                  ('bottom_right', tuple(trapeze.br)),
                  ('bottom_left', tuple(trapeze.bl)),
                  ('size', trapeze_size(trapeze)),
               ))

def trapeze_to_box(trapeze):
    return ( min(trapeze.tl.x, trapeze.bl.x),
             min(trapeze.tr.y, trapeze.tr.y),
             max(trapeze.tr.x, trapeze.br.x),
             max(trapeze.bl.y, trapeze.br.y) 
           )


    
    
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
    
    
def _get_line_equation(point_1, point_2):
    ''' 
        Returns the function that correspond to the 
        equation of the line with those two points
    '''
    # y = mx + b
    try:
        m = (point_2.y - point_1.y) / (point_2.x - point_1.x)
    except Exception:
        print('point_1 '+ str(point_1), 'point_2 ' + str(point_2))
    
    b = point_1.y - m*point_1.x
    def func(x=None, y=None):
        if x is None and y is None:
            raise ValueError('One of the argumentes must not be null')
        if x is not None:
            return m*x + b
        else:
            return (y-b) / m
    return func
       
       
def _generate_wall_coordinates(vanishing_point, x_distance_from_center,
                               top_left, top_right, bottom_right, bottom_left):
    """
        Generates the trapeze to a new wall.
         - vanishing_point: The vanishing point to be used
         - x_distance_from_center: the distance that the right and left 
                                   borders of *the new image* will have
                                   with the center of the image in the
                                   x axix
         - The rest: The corners of the original image.

        Returns: Trapeze 
    """
    center_x = top_right.x - (top_right.x - top_left.x) / 2
    to_x_left = center_x - x_distance_from_center
    to_x_right = center_x + x_distance_from_center
    
    # top left corner
    f_tl = _get_line_equation(top_left, vanishing_point)
    y_tl = f_tl(x = to_x_left)
    tl_result = Point(to_x_left, y_tl)
    
    # top right corner
    f_tr = _get_line_equation(top_right, vanishing_point)
    y_tr = f_tr(x = to_x_right)
    tr_result = Point(to_x_right, y_tr)
    
    # bottom left corner
    f_bl = _get_line_equation(bottom_left, vanishing_point)
    y_bl = f_bl(x = to_x_left)
    bl_result = Point(to_x_left, y_bl)
    
    # bottom right corner
    f_br = _get_line_equation(bottom_right, vanishing_point)
    y_br = f_br(x = to_x_right)
    br_result = Point(to_x_right, y_br)
    
    t = Trapeze(tl_result, tr_result, br_result, bl_result)
    return t
    
 
def _generate_corridor_coordinates(vanishing_point, from_top_point, from_bottom_point, to_x):
    """
        Generates a 'side wall'.

        Explanation of the arguments:
            - In the case of a 'left wall':

                |\
                | \
                |  * -> from_top_point
                |  |
                |  * -> from_bottom_point
                | /
                |/
                    to_x -> the left edge of the wall
    """
    # to top points
    f_t = _get_line_equation(from_top_point, vanishing_point)
    y_t = f_t(x = to_x)
    
    
    # to bottom points
    f_b = _get_line_equation(from_bottom_point, vanishing_point)
    y_b = f_b(x = to_x)
    
    if from_top_point.x > to_x:
        # left corridor wall
        ul_result = Point(to_x, y_t)
        bl_result = Point(to_x, y_b)
        ur_result = from_top_point
        br_result = from_bottom_point
    else:
        # right corridor wall
        ur_result = Point(to_x, y_t)
        br_result = Point(to_x, y_b)
        ul_result = from_top_point
        bl_result = from_bottom_point
        
    return Trapeze(ul_result, ur_result, br_result, bl_result)
    
 
def generate_tiles(source_wall_filename, result_filename, depth, 
                   vanishing_point_offset=(0,0), new_size=None, source_offset=(0,0), crop=False):
    """
        Main function, that creates all the tiles.

        Input:
            - source_wall_filename:     Filename of the original wall image
            - result_filename:          The base name of the resulting images
            - depth:                    The max depth to generate the tiles. 
                                        The original image is considered depth 
                                        1. The new tiles will be 'ahead' of
                                        the original.
            - vanishing_point_offset:   By default, the vanishing point will be
                                        in the center of the image. The offset
                                        displaces it is needed.
            - new_size:                 The size of the new images, if you want 
                                        it to be different from the original.
            - source_offset:            By default, the source image is to be 
                                        centered in the new ones. The offset
                                        displaces it is needed.
            - crop:                     So the resulting images are cropped.

        Result:
            - This function has no return.
            - It creates a folder with the resulting images.
            - It also generate a JSON file with data about the generated image.
              It stores the 'new_size', and the corners and size of each generated 
              wall.
    """
    
    source_image = Image.open(source_wall_filename)
    
    # setup
    if new_size and new_size != source_image.size:
        result_image = Image.new('RGBA', new_size)
        
        # calculate source boundaries inside the result
        src_half_x = source_image.size[0] / 2
        src_half_y = source_image.size[1] / 2
        res_half_x = result_image.size[0] / 2
        res_half_y = result_image.size[1] / 2
        source_tl = Point(res_half_x - src_half_x + source_offset[0],
                          res_half_y - src_half_y + source_offset[0])
        source_tr = Point(res_half_x + src_half_x + source_offset[0],
                          res_half_y - src_half_y + source_offset[0])
        source_br = Point(res_half_x + src_half_x + source_offset[0],
                          res_half_y + src_half_y + source_offset[0])
        source_bl = Point(res_half_x - src_half_x + source_offset[0],
                          res_half_y + src_half_y + source_offset[0])
        source_box = Trapeze(source_tl, source_tr, source_br, source_bl)
        
        # paste source image into new image
        tmp = tuple( int(x) for x in source_box.tl) # hack cast to int
        result_image.paste(source_image, tmp)
        
    else:
        result_image = source_image
        source_box = Trapeze(Point(0,0),
                             Point(source_image.size[0], 0),
                             Point(source_image.size[0], source_image.size[1]),
                             Point(0, source_image.size[1]))
        
    vanishing_point = Point(result_image.size[0]/2 + vanishing_point_offset[0],
                            result_image.size[1]/2 + vanishing_point_offset[1])
    part = source_image.size[0] / (depth * 2)
    
    result_trapezes = {}

    # near tile
    result_trapezes['n'] = source_box
    # near left tile
    left_t = _generate_corridor_coordinates(vanishing_point, 
                                            source_box.tl,
                                            source_box.bl,
                                            (result_image.size[0] - source_image.size[0])/2 - part)
    result_trapezes['n_l'] = left_t
    # near right tile
    right_t = _generate_corridor_coordinates(vanishing_point, 
                                             source_box.tr,
                                             source_box.br,
                                             (result_image.size[0] + source_image.size[0])/2 + part)
    result_trapezes['n_r'] = right_t
                
    for j in range(depth-1, 0, -1): 
        # creating the 'front walls'
        front_t = _generate_wall_coordinates(vanishing_point, (j)*part, *source_box)
        result_trapezes['f'*(depth-j)] = front_t
        
        # creating the 'side walls' (corridor)
        t_size = trapeze_size(front_t)
        for i in count(0):
            no_left = True
            no_right = True
            if (result_image.size[0] - t_size[0])/2 - t_size[0]*i >= 0:
                # corridor on the left
                tl = Point(front_t.tl[0] - t_size[0]*i, front_t.tl[1])
                bl = Point(front_t.bl[0] - t_size[0]*i, front_t.bl[1])
        
                left_t = _generate_corridor_coordinates(vanishing_point, 
                                                        tl, 
                                                        bl, 
                                                        (result_image.size[0] - t_size[0])/2 - t_size[0]*i - part*(i+1))
                result_trapezes['f'*(depth-j) + '_' + 'l'*(i+1)] = left_t
                
                no_left = False
            if (result_image.size[0] + t_size[0])/2 + t_size[0]*i < result_image.size[0]:
                # corridor to the right
                tr = Point(front_t.tr[0] + t_size[0]*i, front_t.tr[1])
                br = Point(front_t.br[0] + t_size[0]*i, front_t.br[1])
                right_t = _generate_corridor_coordinates(vanishing_point, 
                                                         tr,
                                                         br, 
                                                         (result_image.size[0] + t_size[0])/2 + t_size[0]*i + part*(i+1))
                result_trapezes['f'*(depth-j) + '_' + 'r'*(i+1)] = right_t
                no_right = False
            if no_left and no_right:
                break
        
    # preparing to save stuff
    save_dir = os.path.join(os.getcwd(), result_filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_images = {}
    result_json = { 'image_size' : result_image.size,
                         'tiles' : {}
                       }
    
    for k in result_trapezes.keys(): 
        coeffs = _find_coeffs(list(result_trapezes[k]), list(source_box))
        new_image = result_image.transform(result_image.size, 
                                           Image.PERSPECTIVE,
                                           coeffs,
                                           Image.BICUBIC)
        if crop:
            box = trapeze_to_box(result_trapezes[k])
            box = tuple(round(x) for x in box)
            new_image = new_image.crop(box)

        tile_filename = '{0}_{1}.png'.format(result_filename, k) 
        new_image.save(os.path.join(save_dir, tile_filename), 'PNG')
        result_json['tiles'][tile_filename] = trapeze_to_dict(result_trapezes[k])

    json_file = os.path.join(save_dir, result_filename + '.json')
    with open(json_file, 'w') as result_json_file:
        json.dump(result_json, result_json_file, sort_keys=True, indent=4)



if __name__ == '__main__':
    #generate_tiles('wall.png' 'cenas', 5, new_size=(256, 192))
    generate_tiles('wall.png', 'cenas', 4, vanishing_point_offset=(0, -30), new_size=(288, 216))
    #generate_tiles('wall.png', 'cenas', 4, new_size=(288, 216), crop=True)
    #generate_tiles('wall.png', 'cenas', 4, new_size=(320, 240))
    #generate_tiles('wall.png', 'cenas', 5, new_size=(512, 288))
    