import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# graph first level vertex class
class Vertex:
    def __init__(self, n, neighbors, position):
        self.position = position
        self.name = n
        self.neighbors = neighbors
        self.visited = False
        self.slopes = {}
        self.merge_node = None


class CollisionNode(Vertex):
    def __init__(self, n, neighbors, position, nearest_cross_node, c_collision=False):
        self.nearest_cross_node = nearest_cross_node
        self.c_collision = c_collision
        super().__init__(n, neighbors, position)


class FinalNode:
    def __init__(self, first_coll, last_coll, c_node, position):
        self.first_coll = first_coll
        self.last_coll = last_coll
        self.nearest_c_node = c_node
        self.position = position


def create_graph(image_mat):
    vertex_list = []
    for i in range(image_mat.shape[0]):
        for j in range(image_mat.shape[1]):
            if image_mat[i][j] >= .5:
                vertex_list.append((i, j))
    return vertex_list


# find vertices neighbors and score them for finding cross nodes and end nodes
def create_score_adjacency_list(ver_list):
    score_adjacency_list = []
    for i in range(len(ver_list)):
        score = 0
        indexes = []
        first_levels = 4*[False]
        vertex = ver_list[i]
        if (vertex[0]+1, vertex[1]) in ver_list:
            score += 2
            first_levels[2] = True
            indexes.append(ver_list.index((vertex[0] + 1, vertex[1])))
        if (vertex[0] - 1, vertex[1]) in ver_list:
            score += 2
            first_levels[0] = True
            indexes.append(ver_list.index((vertex[0] - 1, vertex[1])))
        if (vertex[0], vertex[1] + 1) in ver_list:
            score += 2
            first_levels[1] = True
            indexes.append(ver_list.index((vertex[0], vertex[1] + 1)))
        if (vertex[0], vertex[1] - 1) in ver_list:
            score += 2
            first_levels[3] = True
            indexes.append(ver_list.index((vertex[0], vertex[1] - 1)))
        if (vertex[0] + 1, vertex[1] + 1) in ver_list:
            if not(first_levels[1] or first_levels[2]):
                score += 2
                indexes.append(ver_list.index((vertex[0] + 1, vertex[1] + 1)))
        if (vertex[0] + 1, vertex[1] - 1) in ver_list:
            if not(first_levels[2] or first_levels[3]):
                score += 2
                indexes.append(ver_list.index((vertex[0] + 1, vertex[1] - 1)))
        if (vertex[0] - 1, vertex[1] + 1) in ver_list:
            if not(first_levels[0] or first_levels[1]):
                score += 2
                indexes.append(ver_list.index((vertex[0] - 1, vertex[1] + 1)))
        if (vertex[0] - 1, vertex[1] - 1) in ver_list:
            if not(first_levels[0] or first_levels[3]):
                score += 2
                indexes.append(ver_list.index((vertex[0] - 1, vertex[1] - 1)))
        score_adjacency_list.append((score, indexes))
    return score_adjacency_list


def get_vertices(g):
    adjacency = create_score_adjacency_list(g)
    g_end_nodes, g_cross_nodes = [], []
    # create a dictionary of vertices which key is vertex name and value is a Vertex
    g_vertices = {}
    for i in range(len(adjacency)):
        if adjacency[i][0] > 5:
            g_cross_nodes.append(i)
        elif len(adjacency[i][1]) == 1:
            g_end_nodes.append(i)
        adjacency[i] = adjacency[i][1]
        # vertex created
        g_vertices[i] = Vertex(i, adjacency[i], g[i])
    g_end_nodes, g_cross_nodes = prune(g_end_nodes, g_cross_nodes, g_vertices)
    return g_vertices, g_end_nodes, g_cross_nodes


def get_angel(point_pos, center):
    x = point_pos[1] - center[1]
    s_x = np.sign(x)
    y = point_pos[0] - center[0]
    s_y = np.sign(y)
    tetha = math.degrees(math.atan(x/y))
    if s_x == -1:
        tetha += 180
    elif s_y == -1:
        tetha += 360
    return tetha


def get_farthest_vector_index(target_key, target_value,  my_dict):
    d = dict(my_dict)
    for k, v in d.items():
        s = (d[k][1]**2 + d[k][2]**2)**.5
        x = d[k][1]/s
        y = d[k][2]/s
        d[k] = x, y
    t = d[target_key]
    del d[target_key]
    return min(d,  key=lambda i: np.dot(d[i], t))


def get_out_slope(my_dict):
    l = []
    key_list = []
    to_delete = []
    for k, v in my_dict.items():
        key_list.append(k)
        l.append(tuple((k, get_farthest_vector_index(k, v, my_dict))))
    for item in l:
        if item[::-1] in l:
            to_delete.append(item[0])
    l = [x for x in key_list if x not in to_delete]
    return my_dict[l[0]]


def distance(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def get_nodes_mean(n):
    cross_nodes_mean = tuple([sum(x)/len(n) for x in zip(*n)])
    return list(np.round(cross_nodes_mean))


def prune(end_nodes, cross_nodes, verts, prune_size=5):
    to_delete_end = []
    to_delete_cross = []
    for e in end_nodes:
        q = list()
        node = verts[e]
        q.append(node)
        seen = [node.name]
        while len(q) > 0:
            node_u = q.pop()
            seen.append(node_u.name)
            if len(seen) == prune_size:
                break
            for v in node_u.neighbors:
                if v in cross_nodes and v != node.name:
                    to_delete_end.append(e)
                    to_delete_cross.append(v)
                node_v = verts[v]
                if node_v.name not in seen:
                    q.append(verts[v])
    ends = [x for x in end_nodes if x not in to_delete_end]
    crosses = [x for x in cross_nodes if x not in to_delete_cross]
    return ends, crosses


def shift_contour(contour, colls):
    for i in range(len(contour)):
        if contour[i] in colls:
            s = i
            break
    return contour[s:] + contour[:s]


def find_neighbors(position, cnt_vertices):
    for k, v in cnt_vertices.items():
        if v.position == position:
            indx = k
    return cnt_vertices[indx].neighbors


def find_nearest_cross_node(node, skel_cross_nodes, chosen_c_nodes, c_groups, verts):
    q = list()
    q.append(verts[node])
    seen = [node]
    while len(q) > 0:
        node_u = q.pop()
        seen.append(node_u.name)
        for v in node_u.neighbors:
            if v in skel_cross_nodes and v != node:
                adjacent_cross_node = v
                break
            node_v = verts[v]
            if node_v.name not in seen:
                q.append(verts[v])
    for i in c_groups:
        if adjacent_cross_node in i:
            for c in chosen_c_nodes:
                if c in i:
                    return c


def merge_lists_with_common_items(l):
    out = []
    while len(l) > 0:
        first, *rest = l
        first = set(first)
        lf = -1
        while len(first) > lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2
        out.append(list(first))
        l = rest
    return out


def get_cross_nodes_position(c_nodes_batch, verts):
    cross_nodes_dict = {}
    for x in c_nodes_batch:
        cross_nodes_dict[x[0]] = get_nodes_mean([verts[c].position for c in x])
    return cross_nodes_dict


def get_fitted_line_slope(points_list, my_node, verts):
    x = np.array([verts[s].position[1] for s in points_list])
    y = np.array([verts[s].position[0] for s in points_list])
    z = np.polyfit(x, y, 1)
    delta_x = (verts[points_list[-1]].position[1] - my_node.position[1]) / len(points_list)
    delta_y = (verts[points_list[-1]].position[0] - my_node.position[0]) / len(points_list)
    return z, delta_x, delta_y


def get_cross_node_direction(node, c_nodes, e_nodes, verts):
    seen_cross_neighbors = []
    cross_nodes_group = [node.name]
    to_merged_distance = 15
    for i in list(set(node.neighbors).difference(seen_cross_neighbors)):
        q = list()
        q.append(verts[i])
        seen_cross_neighbors.append(i)
        seen = [node.name]
        while len(q) > 0:
            node_u = q.pop()
            seen.append(node_u.name)
            if len(seen) == to_merged_distance or node_u.name in e_nodes:
                node.slopes[i] = get_fitted_line_slope(seen[:10], node, verts)
                break
            for v in node_u.neighbors:
                if v in c_nodes and v != node.name and v not in cross_nodes_group:
                    cross_nodes_group.append(v)
                    break
                node_v = verts[v]
                if node_v.name not in seen:
                    q.append(verts[v])
    if len(cross_nodes_group) != 1 or len(node.slopes)%2 == 0:
        return cross_nodes_group
    if len(node.slopes)%2 == 0:
        print('WARNING! THERE ARE EVEN DIRECTIONS')
    return get_out_slope(node.slopes)


def get_end_node_direction(node, verts):
    q = list()
    seen = [node.name]
    for v in node.neighbors:
        q.append(verts[v])
    while len(q) > 0:
        node_u = q.pop()
        seen.append(node_u.name)
        if len(seen) == 6:
            node.slopes = get_fitted_line_slope(seen, node, verts)
            break
        for v in node_u.neighbors:
            node_v = verts[v]
            if node_v.name not in seen:
                q.append(verts[v])
    return node.slopes


def get_position(dirr, node_pos, contour):
    flag = 0
    x, y = node_pos[1], node_pos[0]
    p = np.poly1d(dirr[0])
    if dirr[1]:
        while node_pos not in contour:
            x = x - dirr[1]
            y = p(x)
            node_pos = np.round(y), np.round(x)
            new_flag = np.sign(cv2.pointPolygonTest(np.array(contour), node_pos, True))
            if flag * new_flag == -1:
                node_pos = min(contour, key=lambda x: distance(x, node_pos))
                break
            flag = new_flag
    else:
        while node_pos not in contour:
            y = y - dirr[2]
            node_pos = np.round(y), x
    return node_pos


def get_collision_with_contour(e_nodes, c_nodes, contour, verts, contour_verts):
    collision_points = []
    c_nodes_groups = []
    for c in c_nodes:
        pos = verts[c].position
        direction = get_cross_node_direction(verts[c], c_nodes, e_nodes, verts)
        if type(direction) == list:
            c_nodes_groups.append(direction)
        if type(direction) == tuple:
            c_nodes_groups.append([c])
            pos = get_position(direction, pos, contour)
            collision_points.append(CollisionNode(str(c) + str(int(pos[0])) + str(int(pos[1])),
                                                  find_neighbors(pos, contour_verts), pos, c, True))
    c_nodes_groups = merge_lists_with_common_items(c_nodes_groups)
    chosen_nodes = [x[0] for x in c_nodes_groups]
    for e in e_nodes:
        pos = verts[e].position
        direction = get_end_node_direction(verts[e], verts)
        pos = get_position(direction, pos, contour)
        collision_points.append(CollisionNode(str(e)+str(int(pos[0]))+str(int(pos[1])),
                                find_neighbors(pos, contour_verts), pos, find_nearest_cross_node(
                                                e, c_nodes, chosen_nodes, c_nodes_groups, verts)))
    return collision_points, c_nodes_groups


def get_other_neighbor(node, all_nodes):
    for n in all_nodes:
        if n.nearest_c_node == node.nearest_c_node and n.last_coll != n.first_coll:
            return n


def get_vertex_by_position(pos, cnt_verts):
    for _, v in cnt_verts.items():
        if v.position == pos:
            return v


def get_to_replace(all_nodes, verts, cnt_verts):
    to_replace_nodes = []
    for n in all_nodes:
        if n.first_coll == n.last_coll:
            n_neighbor = get_other_neighbor(n, all_nodes)
            neighbor_vect = n_neighbor.position[1] - n.position[1], n_neighbor.position[0] - n.position[0]
            len_neighbor_vect = (neighbor_vect[0]**2 + neighbor_vect[1]**2)**.5
            c_node_vect = verts[n.nearest_c_node].position[1] - n.position[1], verts[n.nearest_c_node].position[0] - \
                          n.position[0]
            len_c_node_vect = (c_node_vect[0]**2 + c_node_vect[1]**2)**.5
            dot_product = sum(list(l * r for l, r in zip(neighbor_vect, c_node_vect)))/(len_c_node_vect*len_neighbor_vect)

            to_replace_nodes.append((get_vertex_by_position(n.position, cnt_verts),
                                     np.round(math.sin(math.acos(dot_product))*len_neighbor_vect), n.nearest_c_node))
    return to_replace_nodes


def get_replace_nodes(to_replace, cnt_verts):
    node = to_replace[0]
    width = to_replace[1]-1
    c_node = to_replace[2]
    new_nodes = []
    seen_neighbors = []
    for i in list(set(node.neighbors).difference(seen_neighbors)):
        q = list()
        q.append(cnt_verts[i])
        seen_neighbors.append(i)
        seen = [node.name]
        while len(q) > 0:
            node_u = q.pop()
            seen.append(node_u.name)
            if len(seen) == width:
                new_nodes.append(FinalNode(1, 1, c_node, node_u.position))
                break
            for v in node_u.neighbors:
                node_v = cnt_verts[v]
                if node_v.name not in seen:
                    q.append(cnt_verts[v])
    return new_nodes


def get_final_nodes(verts, cnt_verts, all_nodes):
    to_replaces = get_to_replace(all_nodes, verts, cnt_verts)
    to_del = [n for n in all_nodes if n.first_coll == n.last_coll]
    all_nodes = [n for n in all_nodes if n not in to_del]
    for i in to_replaces:
        all_nodes += get_replace_nodes(i, cnt_verts)
    return all_nodes


def add_inner_contours(all_nodes, all_contours, verts):
    c_nodes = [n.nearest_c_node for n in all_nodes]
    nearest_positions = {}
    for c in set(c_nodes):
        for cnt in all_contours[1:]:
            pos = min(cnt, key=lambda x: distance(x, verts[c].position))
            if c not in nearest_positions:
                nearest_positions[c] = [(pos, distance(pos, verts[c].position))]
            else:
                nearest_positions[c].append((pos, distance(pos, verts[c].position)))
    for _, v in nearest_positions.items():
        v.sort(key=lambda tup: tup[1])

    for c in set(c_nodes):
        for i in range(4 - c_nodes.count(c)):
            if len(nearest_positions[c]) != 0:
                all_nodes.append(FinalNode(1, 1, c, nearest_positions[c][0][0]))
                nearest_positions[c] = nearest_positions[c][1:]

    return all_nodes


def get_sort_points(points, center):
    points_angel = []
    for point in points:
        points_angel.append((point, get_angel(point, center)))
    sorted_points = sorted(points_angel, key=lambda tup: tup[1], reverse=True)
    return [x[0] for x in sorted_points]


def get_collision_point_by_position(pos, coll_points):
    for p in coll_points:
        if p.position == pos:
            return p


def get_nodes(colls, mean_points, contour):
    indexes = []
    for i in range(len(contour)):
        if contour[i] in collision_positions:
            indexes.append((i, contour[i]))
    indexes.append(indexes[0])
    indexes_coll_points = [get_collision_point_by_position(i[1], colls) for i in indexes]
    closest_nodes = []
    for i in range(len(indexes) - 1):
        if i == len(indexes) - 2:
            curr_c_node = indexes_coll_points[i].nearest_cross_node
            curr_c_collision = indexes_coll_points[i].c_collision
            next_c_node = indexes_coll_points[0].nearest_cross_node
            next_c_collision = indexes_coll_points[0].c_collision
            current_range = contour[indexes[i][0]:-1]
        else:
            curr_c_node = indexes_coll_points[i].nearest_cross_node
            curr_c_collision = indexes_coll_points[i].c_collision
            next_c_node = indexes_coll_points[i + 1].nearest_cross_node
            next_c_collision = indexes_coll_points[i + 1].c_collision
            current_range = contour[indexes[i][0]: indexes[i + 1][0]]
        if curr_c_node == next_c_node:
            if curr_c_collision is True:
                closest_nodes.append(FinalNode(indexes_coll_points[i].name, indexes_coll_points[i].name,
                                               curr_c_node, indexes_coll_points[i].position))
            elif next_c_collision is False:
                nearest_pos = min(current_range, key=lambda x: distance(x, mean_points[curr_c_node]))
                closest_nodes.append(
                    FinalNode(indexes_coll_points[i].name, indexes_coll_points[i+1].name, curr_c_node, nearest_pos))
        else:
            if curr_c_collision is True:
                if next_c_collision is False:
                    closest_nodes.append(FinalNode(indexes_coll_points[i].name, indexes_coll_points[i].name,
                                                   curr_c_node, indexes_coll_points[i].position))
                    nearest_pos2 = min(current_range, key=lambda x: distance(x, mean_points[next_c_node]))
                    closest_nodes.append(
                        FinalNode(indexes_coll_points[i].name, indexes_coll_points[i + 1].name, next_c_node, nearest_pos2))
                else:
                    closest_nodes.append(FinalNode(indexes_coll_points[i].name, indexes_coll_points[i].name,
                                                   curr_c_node, indexes_coll_points[i].position))
            else:
                if next_c_collision is True:
                    nearest_pos1 = min(current_range, key=lambda x: distance(x, mean_points[curr_c_node]))
                    closest_nodes.append(
                        FinalNode(indexes_coll_points[i].name, indexes_coll_points[i + 1].name, curr_c_node,
                                  nearest_pos1))
                else:
                    nearest_pos1 = min(current_range, key=lambda x: distance(x, mean_points[curr_c_node]))
                    nearest_pos2 = min(current_range, key=lambda x: distance(x, mean_points[next_c_node]))
                    closest_nodes.append(
                        FinalNode(indexes_coll_points[i].name, indexes_coll_points[i + 1].name, curr_c_node, nearest_pos1))
                    closest_nodes.append(
                        FinalNode(indexes_coll_points[i].name, indexes_coll_points[i + 1].name, next_c_node, nearest_pos2))

    return closest_nodes


def get_average_points(f_nodes):
    my_nodes = []
    for n in f_nodes:
        my_nodes.append((n.nearest_c_node, n.position))
    d = {}
    for tp in my_nodes:
        key, val = tp
        d.setdefault(key, []).append(val)
    for name, values in d.items():
        avg = [sum(x)/len(values) for x in zip(*values)]
        d[name] = (avg[0], avg[1])
    return d

def color_pixel(image, pos, color):
    image[pos[0], pos[1]] = color
    image[pos[0]+1, pos[1]+1] = color
    image[pos[0]+1, pos[1]-1] = color
    image[pos[0]-1, pos[1]+1] = color
    image[pos[0]-1, pos[1]-1] = color

    image[pos[0], pos[1]+1] = color
    image[pos[0], pos[1]-1] = color
    image[pos[0]+1, pos[1]] = color
    image[pos[0]-1, pos[1]] = color
    return image


def plot_overlap_org_img(overlap_poses, org_img):
    org_img = np.dstack([org_img, org_img, org_img])
    for i in range(org_img.shape[0]):
        for j in range(org_img.shape[1]):
            for c in contours:
                if (i, j) in c:
                    if (i, j) in overlap_poses:
                        org_img[i, j] = [1, 0, 0]
                        # org_img = color_pixel(org_img, (i, j), [1, 0, 0])
                    elif list(org_img[i, j]) != [1, 0, 0]:
                        org_img[i, j] = [0, 0, 1]
    plt.imshow(org_img)
    plt.show()
    plt.close()
    plt.imsave(f'output/org-{name}-overlap.bmp', org_img, cmap='Greys')


def plot_overlap_contour(overlap_poses):
    s = np.zeros(img.shape)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            for c in contours:
                if (i, j) in c:
                    if (i, j) in overlap_poses:
                        s[i, j] = 2
                    else:
                        s[i, j] = 1
    plt.imshow(s + img)
    plt.show()
    plt.close()
    plt.imsave(f'output/contour-{name}-overlap.bmp', s)
