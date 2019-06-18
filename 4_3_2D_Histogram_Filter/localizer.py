# import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []
    "def sense(p, Z):\n",
    "    ''' Takes in a current probability distribution, p, and a sensor reading, Z.\n",
    "        Returns a *normalized* distribution after the sensor measurement has been made, q.\n",
    "        This should be accurate whether Z is 'red' or 'green'. '''\n",
    "    q=[]\n",
    "    # loop through all grid cells\n",
    "    for i in range(len(p)):\n",
    "        # check if the sensor reading is equal to the color of the grid cell\n",
    "        # if so, hit = 1\n",
    "        # if not, hit = 0\n",
    "        hit = (Z == world[i])\n",
    "        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))\n",
    "        \n",
    "    # sum up all the components\n",
    "    s = sum(q)\n",
    "    # divide all elements of q by the sum to normalize\n",
    "    for i in range(len(p)):\n",
    "        q[i] = q[i] / s\n",
    "    return q\n",
    #
    # TODO - implement this in part 2
    #
    # loop through all grid cells
    height =  len(grid)
    width = len(grid[0])
    #print(grid)
    #print(beliefs)
    sum_p = 0
    for i in range(height):
        temp_list = []
        for j in range(width):
            # check if the sensor reading is equal to the color of the grid cell
            # if so, hit = 1, else hit = 0
            hit = (color == grid[i][j])
            val = beliefs[i][j] * (hit * p_hit + (1-hit) * p_miss)
            sum_p += val
            temp_list.append(val)
        new_beliefs.append(temp_list)
    #print('sum_p = ', sum_p)
    for i in range(height):
        for j in range(width):
            new_beliefs[i][j] = new_beliefs[i][j]/sum_p    
        
    return new_beliefs

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        # i corresponds to height
        for j, cell in enumerate(row):
            # j corresponds to width
            new_i = (i + dy ) % height
            new_j = (j + dx ) % width #height
            # pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = cell
    return blur(new_G, blurring)