import itertools

from scipy import misc
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt

# Load the image from disk as a numpy ndarray


# Create a flat color image for graph building:



# Defines a translation from 2 coordinates to a single number
def to_index(img,y, x):
    return y * img.shape[1] + x


# Defines a reversed translation from index to 2 coordinates
def to_coordinates(img,index):
    return int(index / img.shape[1]), index % img.shape[1]


# A sparse adjacency matrix.
# Two pixels are adjacent in the graph if both are painted.
def main(original_img,pathimage,x1,x2,y1,y2):
    img = original_img[:, :, 0] + original_img[:, :, 1] + original_img[:, :, 2]
    adjacency = dok_matrix((img.shape[0] * img.shape[1],
                            img.shape[0] * img.shape[1]), dtype=bool)

# The following lines fills the adjacency matrix by
    directions = list(itertools.product([0, 1, -1], [0, 1, -1]))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if not img[i, j]:
                continue

            for y_diff, x_diff in directions:
                if img[i + y_diff, j + x_diff]:
                    adjacency[to_index(img,i, j),
                            to_index(img,i + y_diff, j + x_diff)] = True

# We chose two arbitrary points, which we know are connected
    source = to_index(img,y1, x1)
    target = to_index(img,y2, x2)

# Compute the shortest path between the source and all other points in the image
    _, predecessors = dijkstra(adjacency, directed=False, indices=[source],
                             unweighted=True, return_predecessors=True)

# Constructs the path between source and target
    pixel_index = target
    pixels_path = []
    while pixel_index != source:
        pixels_path.append(pixel_index)
        pixel_index = predecessors[0, pixel_index]
        if(pixel_index==-9999):
            return 1


# The following code is just for debugging and it visualizes the chosen path
        
    #original_img.setflags(write=1)
    path=[]
    for pixel_index in pixels_path: 
        i, j = to_coordinates(img,pixel_index)
        print(i,j)
        path.append([i,j])
        pathimage[i, j,0] = 255

    plt.imshow(pathimage)
    plt.show()
    return path