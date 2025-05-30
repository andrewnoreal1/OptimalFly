import math
import sys
import copy
from scipy.spatial import ConvexHull
import numpy as np

def does_line_intersect_polygon(mid, slope, intercept, vertices):
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        
        if x2 != x1:  # non-vertical line
            m_e = (y2 - y1) / (x2 - x1)
            b_e = y1 - m_e * x1
            if slope != m_e:  # ensure lines are not parallel
                x = (b_e - intercept) / (slope - m_e)
                y = slope * x + intercept
                if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and x != mid[0] and y != mid[1]:
                    print(f"Intersection at ({x}, {y}) between ({x1}, {y1}) and ({x2}, {y2})")
                    return x, y

        else:  # handle vertical line case
            x = x1
            y = slope * x + intercept
            if min(y1, y2) <= y <= max(y1, y2) and x != mid[0] and y != mid[1]:
                print(f"Intersection at ({x}, {y}) on vertical segment ({x1}, {y1}) to ({x2}, {y2})")
                return x, y


def haversine(lat1, lon1, lat2, lon2):
    R = 6378000  # Earth's radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def convert_to_cartesian(positions):
    # Find the minimum latitude and longitude to use as the origin
    min_lat = min(positions, key=lambda x: x[0])[0]
    min_lon = min(positions, key=lambda x: x[1])[1]

    # Create a list to hold Cartesian coordinates
    cartesian_coords = []

    # Convert each position to Cartesian coordinates
    for lat, lon in positions:
        # Distance from the minimum longitude to the current point's longitude (x-coordinate)
        x = haversine(min_lat, min_lon, min_lat, lon)
        # Distance from the minimum latitude to the current point's latitude (y-coordinate)
        y = haversine(min_lat, min_lon, lat, min_lon)

        # Append the calculated Cartesian coordinates to the list
        cartesian_coords.append((x, y))

    return cartesian_coords

def calculate_polygon_edges(positions):
    """
    Calculate all the edges of a polygon and return their lengths.
    
    :param positions: A list of tuples [(lat, lon), ...] representing the polygon vertices.
    :return: A list of distances for each edge of the polygon.
    """
    num_vertices = len(positions)
    distances = []
    
    # Iterate through each vertex and calculate distance to the next vertex
    for i in range(num_vertices):
        # Get current vertex and the next vertex, wrapping around to the first vertex
        lat1, lon1 = positions[i]
        lat2, lon2 = positions[(i + 1) % num_vertices]  # Wrap around using modulo
        
        # Calculate the distance between the vertices
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    return distances

def calculate_edge_lengths(cartesian_coords):
    """
    Calculate the lengths of edges in a polygon given its vertices in Cartesian coordinates.
    
    :param cartesian_coords: A list of tuples (x, y) representing the Cartesian coordinates of the polygon vertices.
    :return: A list of lengths of each edge in the polygon.
    """
    num_vertices = len(cartesian_coords)
    edge_lengths = []
    
    for i in range(num_vertices):
        # Current vertex
        x1, y1 = cartesian_coords[i]
        # Next vertex, with wrap-around using modulo to close the polygon
        x2, y2 = cartesian_coords[(i + 1) % num_vertices]
        
        # Calculate the distance between the current vertex and the next vertex
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        edge_lengths.append(distance)
    
    return edge_lengths

def calculate_polygon_area(cartesian_coords):
    """
    Calculate the area of a polygon given its vertices in Cartesian coordinates using the Shoelace theorem.
    
    :param cartesian_coords: A list of tuples (x, y) representing the Cartesian coordinates of the polygon vertices.
    :return: The area of the polygon in square meters.
    """
    n = len(cartesian_coords)  # Number of vertices
    area = 0

    # Sum over the vertices
    for i in range(n):
        j = (i + 1) % n  # Next vertex index, wraps around
        x_i, y_i = cartesian_coords[i]
        x_j, y_j = cartesian_coords[j]
        area += x_i * y_j - y_i * x_j

    area = abs(area) / 2.0
    return area

def find_longest_edge(cartesian_coords):
    """
    Find the vertices that form the longest edge in a polygon given its vertices in Cartesian coordinates.
    
    :param cartesian_coords: A list of tuples (x, y) representing the Cartesian coordinates of the polygon vertices.
    :return: A tuple containing:
        - the length of the longest edge,
        - the coordinates of the start vertex of the longest edge,
        - the coordinates of the end vertex of the longest edge.
    """
    num_vertices = len(cartesian_coords)
    longest_edge_length = 0
    longest_edge_vertices = (None, None)
    
    for i in range(num_vertices):
        # Current vertex
        x1, y1 = cartesian_coords[i]
        # Next vertex, with wrap-around using modulo to close the polygon
        x2, y2 = cartesian_coords[(i + 1) % num_vertices]
        
        # Calculate the distance between the current vertex and the next vertex
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Check if this is the longest edge found so far
        if distance > longest_edge_length:
            longest_edge_length = distance
            longest_edge_vertices = ((x1, y1), (x2, y2))
    
    return longest_edge_length, longest_edge_vertices

def find_midpoint(point1, point2):
    # Extract coordinates from the points
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate the midpoint coordinates
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    
    return (mid_x, mid_y)

def line_equation_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def angle_with_x_axis(slope):
    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def perpendicular_line_equation(midpoint, slope):
    mx, my = midpoint
    if slope == 0:  # Handle horizontal line case
        perp_slope = None  # Undefined slope
        return "x = {}".format(mx)  # Vertical line
    else:
        perp_slope = -1 / slope
        perp_intercept = my - perp_slope * mx
        return perp_slope, perp_intercept
    
def calculate_new_lat_lon(origin_lat, origin_lon, distance_north, distance_east):
    """ Calculate new latitude and longitude from origin given distances north and east. """
    R = 6378000  # Radius of Earth in meters
    delta_lat = distance_north / R  # Change in latitude in radians
    new_lat = origin_lat + math.degrees(delta_lat)  # New latitude in degrees

    # Adjust for change in longitude, which depends on latitude
    r = R * math.cos(math.radians(new_lat))  # Effective radius at new latitude
    delta_lon = distance_east / r  # Change in longitude in radians
    new_lon = origin_lon + math.degrees(delta_lon)  # New longitude in degrees

    return (new_lat, new_lon)

    
def divide_line_into_segments(x1, y1, x2, y2, n):
    points = []
    for i in range(1, n):  # We skip 0 and n because they correspond to the endpoints.
        t = i / n
        xt = (1 - t) * x1 + t * x2
        yt = (1 - t) * y1 + t * y2
        points.append((xt, yt))

    return points

def perpendicular_line_intersect_polygon(slope, intercept, vertices):
    across_points =[]
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        
        if x2 != x1:  # non-vertical line
            m_e = (y2 - y1) / (x2 - x1)
            b_e = y1 - m_e * x1
            if slope != m_e:  # ensure lines are not parallel
                x = (b_e - intercept) / (slope - m_e)
                y = slope * x + intercept
                if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
                    print(f"Intersection at ({x}, {y}) between ({x1}, {y1}) and ({x2}, {y2})")
                    point = (x, y)
                    across_points.append(point)

        else:  # handle vertical line case
            x = x1
            y = slope * x + intercept
            if min(y1, y2) <= y <= max(y1, y2):
                print(f"Intersection at ({x}, {y}) on vertical segment ({x1}, {y1}) to ({x2}, {y2})")
                point = (x, y)
                across_points.append(point)

    return across_points

def divide_points(per_points, polygon, slope1):
    each_point = []
    for point in per_points:
        perp_slope, perp_intercept = perpendicular_line_equation(point, slope1)
        per_dot = perpendicular_line_intersect_polygon(perp_slope, perp_intercept,polygon)

        each_point.extend(per_dot)

    return each_point

# def rotate_matrix (x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
#     """
#     Rotates a point in the xy-plane counterclockwise through an angle about the origin
#     https://en.wikipedia.org/wiki/Rotation_matrix
#     :param x: x coordinate
#     :param y: y coordinate
#     :param x_shift: x-axis shift from origin (0, 0)
#     :param y_shift: y-axis shift from origin (0, 0)
#     :param angle: The rotation angle in degrees
#     :param units: DEGREES (default) or RADIANS
#     :return: Tuple of rotated x and y
#     """

#     # Shift to origin (0,0)
#     x = x - x_shift
#     y = y - y_shift

#     # Convert degrees to radians
#     if units == "DEGREES":
#         angle = math.radians(angle)

#     # Rotation matrix multiplication to get rotated x & y
#     xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
#     yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

#     return xr, yr


def rotate_matrix(x, y, angle, x_shift, y_shift, units="DEGREES", reverse=False):
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin or
    rotates it back (clockwise) if reverse is True.
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param angle: The rotation angle in degrees (by default) or radians n
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param units: DEGREES (default) or RADIANS
    :param reverse: If True, rotate clockwise by the angle, otherwise counterclockwise
    :return: Tuple of rotated x and y
    """

    # Convert degrees to radians if necessary
    if units == "DEGREES":
        angle = math.radians(angle)

    # Reverse the angle if needed
    if reverse:
        angle = -angle

    # Shift the point to the origin (0,0)
    x = x - x_shift
    y = y - y_shift


    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr


def split_area(area, perp, tolerance=1e-6):
    area_list = []

    for i in range(len(perp)):
        one_area =[]
        if i == 0:
            dot = perp[i]
            for point in area: 
                if (abs(dot[1])-abs(point[1]) >= -tolerance ):
                    one_area.append(point)
            
            area_list.append(one_area)
            print(f"MOT{i}")
            print(f"THIS IS AREA")            
            print(f"{one_area}")
        elif i == ((len(perp))-1):
            one_area =[]
            dot = perp[i-1]
            dot2 = perp[i]
            for point in area: 
                if ((abs(point[1])-abs(dot[1])>= -tolerance) and (abs(dot2[1])-abs(point[1])>= -tolerance)):
                    one_area.append(point)
            area_list.append(one_area)        

            one_area =[]
            dot = perp[i]
            for point in area: 
                if (abs(point[1])-abs(dot[1])>= -tolerance):
                    one_area.append(point)
            area_list.append(one_area)
            print(f"HAI{i}")
            print(f"THIS IS AREA")            
            print(f"{one_area}")
        else:
            one_area =[]
            dot = perp[i-1]
            dot2 = perp[i]
            print(f"dot{dot} dot2 {dot2}")
            for point in area: 
                if ((abs(point[1])-abs(dot[1]) >= -tolerance) and (abs(dot2[1])-abs(point[1]) >= -tolerance)):
                    one_area.append(point)
            area_list.append(one_area)
            print(f"BA{i}")
            print(f"THIS IS AREA")            
            print(f"{one_area}")
        print(f"THIS IS LIST")            
        print(f"{area_list}")
        
    return area_list
            


    

def main():

    # Example list of positions forming a polygon
    # positions = [
    #     (21.021573650653128, 105.81055679611137),
    #     (21.024618113575197, 105.81497707656791),
    #     (21.025379219590956, 105.82613506606987),
    #     (21.018368885677415, 105.83141365341118),
    #     (21.01444295471698, 105.8263925581353),
    #     (21.016005327377822, 105.81467666915825)
    # ]

    # positions = [
    #     (21.064750464176473, 105.79298381188937),
    #     (21.06479551762887, 105.7936007199628),
    #     (21.0641672710334, 105.79361681321689),
    #     (21.06417477997252, 105.79305086711474)

    # ]
    positions = [
        (21.038316357324614, 105.78094978776994),
        (21.039828411987216, 105.7824732824904),
        (21.038576712201465, 105.78526277986589),
        (21.036894411121935, 105.78499911346898),
        (21.03700456236909, 105.78248856583104)

        ]
    


    min_lat = min(positions, key=lambda x: x[0])[0]
    min_lon = min(positions, key=lambda x: x[1])[1]
    print(f"{min_lat}")
    print(f"{min_lon}")

    # Convert the geographic positions to Cartesian coordinates
    cartesian_coordinates = convert_to_cartesian(positions)
    print("Cartesian Coordinates:")
    for coord in cartesian_coordinates:
        print(coord)

    distance_cartesian = calculate_edge_lengths(cartesian_coordinates)
    print("\nCartesian distance:")
    for coord in distance_cartesian:
        print(coord)

    distance = calculate_polygon_edges(positions)
    print("\nCoordinate distance:")
    for coord in distance:
        print(coord)

    area = calculate_polygon_area(cartesian_coordinates)
    print(f"\nArea: {area}")

    division_area = area/3

    longest, longest_edge_point = find_longest_edge(cartesian_coordinates)
    print(f"\nEdge: {longest}")
    for coord in longest_edge_point:
        print(coord)
    
    midpoint = find_midpoint(longest_edge_point[0], longest_edge_point[1])
    print(f"\nMidpoint: {midpoint}")
    new = calculate_new_lat_lon(min_lat,min_lon,midpoint[1],midpoint[0])
    print(f"\nGPS Midpoint: {new}")

    slope, intercept = line_equation_from_points(longest_edge_point[0], longest_edge_point[1])
    angle = angle_with_x_axis(slope)
    print(f"\nAngle: {angle}")
    perp_slope, perp_intercept = perpendicular_line_equation(midpoint, slope)
    print(f"\nPerp_slope: {perp_slope},Perp_intercep: {perp_intercept}")

    intersect_point = does_line_intersect_polygon(midpoint,perp_slope, perp_intercept,cartesian_coordinates)
    print(f"\nIntersect: {intersect_point[0]}:{intersect_point[1]}")
    new = calculate_new_lat_lon(min_lat,min_lon,intersect_point[1],intersect_point[0])
    print(f"\nGPS Intersect: {new}")

    position_list1 =[]
    position_list1 = copy.deepcopy(cartesian_coordinates)
    
    position_list1.append(midpoint)
    position_list1.append(intersect_point)
    print(f"{position_list1}")

    
    GPS_list = []
    for point in position_list1:
        new = calculate_new_lat_lon(min_lat,min_lon,point[1],point[0])
        GPS_list.append(new)
        print(f"{new}")
    
    with open('area.txt', 'w') as file:
        for pos in GPS_list:
            file.write(f"{pos[0]}, {pos[1]}\n")
    
    perpendicular_points = divide_line_into_segments(midpoint[0], midpoint[1], intersect_point[0], intersect_point[1],3)

    per_GPS_list = []
    for point in perpendicular_points:
        new = calculate_new_lat_lon(min_lat,min_lon,point[1],point[0])
        per_GPS_list.append(new)
        print(f"{new}")
    with open('per.txt', 'w') as file:
        for pos in per_GPS_list:
            file.write(f"{pos[0]}, {pos[1]}\n")

    div_points = divide_points(perpendicular_points, cartesian_coordinates, perp_slope)
    div_GPS_list = []
    for point in div_points:
        new = calculate_new_lat_lon(min_lat,min_lon,point[1],point[0])
        div_GPS_list.append(new)
        print(f"{new}")
        
    with open('div.txt', 'w') as file:
        for pos in div_GPS_list:
            file.write(f"{pos[0]}, {pos[1]}\n")

    rotated_div_points = []
    for point in div_points:
        new_point = rotate_matrix(point[0], point[1],(-angle), midpoint[0], midpoint[1])
        rotated_div_points.append(new_point)
    
    rotated_perpendicular_points = []
    for point in perpendicular_points:
        new_point = rotate_matrix(point[0], point[1],(-angle), midpoint[0], midpoint[1])
        rotated_perpendicular_points.append(new_point)

    rotated_cartesian_coordinates = []
    for point in cartesian_coordinates:
        new_point = rotate_matrix(point[0], point[1],(-angle), midpoint[0],midpoint[1])
        rotated_cartesian_coordinates.append(new_point)

    rotate_polygon = []
    rotate_polygon = rotated_div_points + rotated_cartesian_coordinates
    print(f"FINAL")
    print(f"{rotate_polygon}")
    print(f"PERPENDICULAR")
    print(f"{rotated_perpendicular_points}")

    rotated_area = split_area(rotate_polygon,rotated_perpendicular_points)
    

    for i in range(len(rotated_area)):
        area = rotated_area[i]
        unrotated_area =[]
        print(f"{area}")
        for point in area:
            #convert back in previous coordinate
            new_point = rotate_matrix(point[0], point[1],(-angle), midpoint[0], midpoint[1], reverse = True)
            unrotated_area.append(new_point)
        per_GPS_list = []
        for point in unrotated_area:
            new = calculate_new_lat_lon(min_lat,min_lon,point[1],point[0])
            per_GPS_list.append(new)
            print(f"{new}")
        
        print(f"{per_GPS_list}")
        '''
        # Convert the list of positions to a NumPy array
        points = np.array(per_GPS_list)
        
        # Calculate the convex hull
        hull = ConvexHull(points)
        
        # Extract the vertices of the convex hull
        hull_vertices = points[hull.vertices]
        
        # Convert the vertices back to a list of tuples
        points = [tuple(point) for point in hull_vertices]

        '''
        with open(f'dien tich{i}.txt', 'w') as file:
            for pos in per_GPS_list:
                file.write(f"{pos[0]}, {pos[1]}\n")





if __name__ == "__main__":
    main()
