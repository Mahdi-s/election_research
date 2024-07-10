import pandas as pa
import math
import numpy as np
import networkx as nx
import random
import plotly.graph_objects as go
from ortools.graph import pywrapgraph
import time

def setup(districts, grid_size, district_centers, pops): 
    try:
        tol = 1
        start_nodes = [0] * (grid_size * grid_size)
        for i in range(grid_size * grid_size):
            start_nodes = start_nodes + [i + 1] * districts
        for j in range(districts):
            start_nodes = start_nodes + [(grid_size * grid_size) + 1 + j]
        end_nodes = []
        for i in range(grid_size * grid_size):
            end_nodes = end_nodes + [i + 1]
        for j in range(grid_size * grid_size):
            for k in range(districts):
                end_nodes = end_nodes + [(grid_size * grid_size) + 1 + k]
        end_nodes = end_nodes + [(grid_size * grid_size) + districts + 1] * districts
        total_pop = np.sum(np.array(pops))
        capacities = np.array(pops).flatten().tolist()
        for i in np.array(pops).flatten().tolist():
            for j in range(districts):
                capacities = capacities + [i]
        for i in range(districts):
            capacities = capacities + [int(total_pop / districts) + tol]
        costs = [0] * (grid_size * grid_size)
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(districts):
                    costs = costs + [int((((i - district_centers[k][0]) ** 2) + ((j - district_centers[k][1]) ** 2)))]
        costs = costs + [0] * districts
        supplies = [int(total_pop)] + ((grid_size * grid_size) + districts) * [0] + [-int(total_pop)]
        source = 0
        sink = districts + (grid_size * grid_size) + 1
        print("done with setup")
        return start_nodes, end_nodes, capacities, costs, supplies, source, sink, pops
    except Exception as e:
        print(f"Error in setup function: {e}")
        raise

def optimize(start_nodes, end_nodes, capacities, costs, supplies, source, sink, grid, grid_size):
    try:
        print("starting optimize")
        Block_Assignments = pa.DataFrame(columns=['DIST_NO', 'ASSIGN_POP', 'ACTUAL_POP'])
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for i in range(len(start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], costs[i])
        for i in range(len(supplies)):
            min_cost_flow.SetNodeSupply(i, supplies[i])
        flag = 0    
        if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
            for arc in range(min_cost_flow.NumArcs()):
                if min_cost_flow.Tail(arc) != source and min_cost_flow.Head(arc) != sink:
                    if min_cost_flow.Flow(arc) > 0:
                        if min_cost_flow.Capacity(arc) == min_cost_flow.Flow(arc):
                            grid.nodes[(int(((min_cost_flow.Tail(arc) - 1) / grid_size)), (min_cost_flow.Tail(arc) - 1) % grid_size)]["district"] = min_cost_flow.Head(arc)
                            Block_Assignments.loc[min_cost_flow.Tail(arc)] = [min_cost_flow.Head(arc), min_cost_flow.Flow(arc), min_cost_flow.Capacity(arc)]
                        else:
                            if flag == 0:
                                grid.nodes[(int(((min_cost_flow.Tail(arc) - 1) / grid_size)), (min_cost_flow.Tail(arc) - 1) % grid_size)]["district"] = min_cost_flow.Head(arc)
                                Block_Assignments.loc[min_cost_flow.Tail(arc)] = [min_cost_flow.Head(arc), min_cost_flow.Flow(arc), min_cost_flow.Capacity(arc)]
                                flag = min_cost_flow.Flow(arc)
                            else: 
                                if Block_Assignments.loc[min_cost_flow.Tail(arc), 'ASSIGN_POP'] < min_cost_flow.Flow(arc):
                                    grid.nodes[(int(((min_cost_flow.Tail(arc) - 1) / grid_size)), (min_cost_flow.Tail(arc) - 1) % grid_size)]["district"] = min_cost_flow.Head(arc)
                                    Block_Assignments.loc[min_cost_flow.Tail(arc)] = [min_cost_flow.Head(arc), min_cost_flow.Flow(arc), min_cost_flow.Capacity(arc)]
                                    flag = flag + min_cost_flow.Flow(arc)
                                else:
                                    flag = flag + min_cost_flow.Flow(arc)
                                if flag == min_cost_flow.Capacity(arc):
                                    flag = 0
        else:
            print('There was an issue with the min cost flow input.')
        print("done with optimize")
        return Block_Assignments
    except Exception as e:
        print(f"Error in optimize function: {e}")
        raise

def grid_setup(grid_s, dist, n, p_0, c, r):
    try:
        grid_size = grid_s
        districts = dist
        district_centers = []
        spacing = int(grid_size / districts)
        for j in range(districts):
            district_centers = district_centers + [[(j + 1) * spacing + random.randint(-1, 1), random.randint(0, grid_size - 1)]]
        pops = []
        row = []
        for i in range(grid_size):
            for j in range(grid_size):
                row = row + [1]
            pops = pops + [row]
            row = []
        grid = nx.grid_graph([grid_size, grid_size])
        for node in grid.nodes():
            grid.nodes[node]["population"] = pops[node[0]][node[1]]
        add_party_preference(grid, n, grid_size, p_0, c, r)
        return grid_size, district_centers, pops, grid
    except Exception as e:
        print(f"Error in grid_setup function: {e}")
        raise

def add_party_preference(grid, n, grid_size, p_0, c, r):
    try:
        squares_with_id = []
        all_nodes = list(grid.nodes())
        all_nodes_processed = []
        voter_arr_ind = 0
        row_tracker = 0
        square = []
        voter_arr_ind = 0
        vertical_mover = 0
        for i in range(n):
            if i != 0:
                vertical_mover += r * grid_size
            for m in range(n):
                row_tracker = (c * m) + vertical_mover
                square = []
                for j in range(r):
                    row = all_nodes[row_tracker:row_tracker + c]
                    square = square + row
                    row_tracker += grid_size
                all_nodes_processed += square
                voter_ratios = np.ones(c * r)
                proportion_of_a = math.floor((c * r) * p_0)
                voter_ratios[:proportion_of_a] = 0
                np.random.shuffle(voter_ratios)
                for node in square:    
                    grid.nodes[node]['voter_pref'] = voter_ratios[voter_arr_ind]
                    voter_arr_ind = voter_arr_ind + 1
                squares_with_id.append(square)
                voter_arr_ind = 0
                square = []
        missed_nodes = list(set(all_nodes).difference(all_nodes_processed))
        if missed_nodes:   
            print("missed_nodes = " + str(missed_nodes))
    except Exception as e:
        print(f"Error in add_party_preference function: {e}")
        raise

# Global variables to store the current state
current_grid = None
current_block_assignments = None
current_districts = None

# Create FigureWidgets outside the function
fig_widget = go.FigureWidget()
fig_widget2 = go.FigureWidget()

def create_district_map(grid, block_assignments):
    x_coords, y_coords, colors, district_colors = [], [], [], []
    party_counts = {'blue': 0, 'red': 0}
    district_color_map = {
        1: 'red', 2: 'blue', 3: 'green', 4: 'purple', 5: 'orange',
        6: 'pink', 7: 'yellow', 8: 'brown', 9: 'cyan', 10: 'magenta'
    }

    for node in grid.nodes():
        x, y = node
        x_coords.append(x)
        y_coords.append(y)
        
        color = 'blue' if grid.nodes[node]['voter_pref'] == 0 else 'red'
        colors.append(color)
        party_counts[color] += 1
        
        district = grid.nodes[node]["district"]
        district_color = district_color_map[(district - (len(grid) * len(grid)) - 1) % 10 + 1]
        district_colors.append(district_color)

    return x_coords, y_coords, colors, district_colors, party_counts



def find_win_count(grid, dist_num):
    win_count = {}

    # Initialize the win_count dictionary
    for node in grid.nodes():
        district_id = grid.nodes[node]['district']
        if district_id not in win_count:
            win_count[district_id] = {0.0: 0, 1.0: 0}
        if len(win_count) == dist_num:
            break

    # Count votes for each district
    for node in grid.nodes():
        district_id = grid.nodes[node]['district']
        voter_pref = float(grid.nodes[node]["voter_pref"])  # Convert to float
        win_count[district_id][voter_pref] += 1

    return win_count

def find_eg(win_count):
    """
    Inputs: dictionary of win lose counts for parties within each district
    Purpose: find efficiency gap of district assignments
    Output: eg of districts packaged as dictionary
    """



    votes_1 = 0
    votes_0 = 0
    district_data = win_count
    wasted_votes = []
    total_win_a = 0
    total_win_b = 0
    total_votes_state = 0
    imp_flag = 0 #set to 0 means immposiblity has not occured
    state_votes_a = 0
    state_votes_b = 0
    P_a = 0 #state_votes_a/total_votes_state
    P_b = 0
    p_a = 0 #total_win_a/state_count
    p_b = 0
    state_count = 0

    for key in win_count:
        state_count += 1
        votes_1 = win_count[key][1.0]
        votes_0 = win_count[key][0.0]
        state_votes_a += votes_0
        state_votes_b += votes_1
        totalVotes = votes_1 + votes_0
        total_votes_state += totalVotes
        
        if votes_1 > votes_0:
            half = math.ceil(totalVotes / 2)
            wasted_1 = votes_1 - half
            imp_flag = 1
            total_win_b += 1
        else:
            wasted_1 = votes_1

        if votes_1 < votes_0:
            half = math.ceil(totalVotes / 2)
            wasted_0 = votes_0 - half
            total_win_a += 1

        else:
            wasted_0 = votes_0
        
        wasted_total = wasted_0 - wasted_1
        
        wasted_votes.append(wasted_total)
        if imp_flag == 1:
            district_data[key]['imp_flag'] = 1
        else:
            district_data[key]['imp_flag'] = 0

    sum_wv =sum(wasted_votes)
    eg = (1/total_votes_state)*sum_wv
    # Implementation of find_eg function (placeholder)
    return {'eg': eg}

def step_five_finder(delta, gamma, dists_num, p_0, n, c, r):
    # Implementation of step_five_finder function
    proportion_a = math.floor((c * r) * p_0)
    proportion_b = (c*r) - proportion_a
    left_side = proportion_b / proportion_a
    F = math.sqrt((1-delta)/(2*dists_num))
    numerator = (F**2)-(4*3.14*gamma**(-1)*F*math.sqrt(2)*(1/n))-(8*((3.14)**2)*(gamma**(-1))*(1/n)**2)
    denominator = (F**2)+(4*3.14*gamma**(-1)*F*math.sqrt(2)*(1/n))+(8*((3.14)**2)*(gamma**(-1))*(1/n)**2)
    right_side = numerator/denominator
    return left_side, right_side

def refined_step_five_finder(delta, gamma, dists_num, p_0, n, c, r):
    # Implementation of refined_step_five_finder function
    proportion_a = math.floor((c * r) * p_0)
    proportion_b = (c*r) - proportion_a
    left_side = proportion_b / proportion_a
    F = math.sqrt((1-delta)/(2*dists_num))
    numerator = 2*(1/n)*(math.pi)
    denominator = (4*gamma)*(math.sqrt((1-delta)/(dists_num+1)))*n
    right_side = numerator/denominator
    return left_side, right_side

def find_winners(win_count):
    party_one_win = 0
    party_zero_win = 0
    
    for district, votes in win_count.items():
        party_zero_count = votes[0.0]
        party_one_count = votes[1.0]
        
        if party_one_count > party_zero_count:
            party_one_win += 1
        elif party_zero_count > party_one_count:
            party_zero_win += 1
        # If counts are equal, we don't increment either win count
    
    return party_one_win, party_zero_win