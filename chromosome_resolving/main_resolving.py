import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

from overlap_resolving import *

name = input()
name = str(name)
path = f'real_overlaps/{name}-'
img = cv2.imread(path + 'skel.bmp', cv2.IMREAD_GRAYSCALE)/255.0
org_img = cv2.imread(path + 'org.bmp', cv2.IMREAD_GRAYSCALE) / 255.0
with open(path + 'contour.txt') as f:
    content = f.readlines()

content = [x.strip() for x in content]
contour_num = int(content[0])
contours = []
end = 1
for i in range(contour_num):
    start = end + 1
    k = int(content[start-1])
    end = start + k
    contours.append([tuple(map(int, x.split(' ')))[::-1] for x in content[start:end]])


skel_graph = create_graph(img)
vertices, end_nodes, cross_nodes = get_vertices(skel_graph)

contour_vertices, _, _ = get_vertices(contours[0])
collisions, cross_nodes_batch = get_collision_with_contour(end_nodes, cross_nodes, contours[0], vertices, contour_vertices)
cross_nodes_positions = get_cross_nodes_position(cross_nodes_batch, vertices)
collision_positions = [x.position for x in collisions]
final_contour = shift_contour(contours[0], collision_positions)


m_point = cross_nodes_positions
if len(m_point) == 0:
    sys.exit()
nodes = get_nodes(collisions, m_point, final_contour)

new_mean = get_average_points(nodes)
while new_mean != m_point:
    nodes = get_nodes(collisions, new_mean, final_contour)
    m_point = new_mean
    new_mean = get_average_points(nodes)

nodes = get_final_nodes(vertices, contour_vertices, nodes)
if len(contours) > 1:
    nodes = add_inner_contours(nodes, contours, vertices)

n_pos = [n.position for n in nodes]
plot_overlap_org_img(n_pos, org_img)
plot_overlap_contour(n_pos)
