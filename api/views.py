from rest_framework.decorators import api_view
from django.http import JsonResponse
import json
import heapq
from collections import deque
import math

@api_view(['POST'])
def Projekat1(request):
    data = json.loads(request.body)
    tiles = data['tiles']
    playerx = data['playerx']
    playery = data['playery']
    player_type = data['player_type']
    gold_positions = data['gold_positions']
    width = 6
    height = 6
    g = Graph(height, width)

    start = (playerx, playery)

    matrix = g.make_matrix(height, width, tiles)
    

    if player_type == 'Aki':
        path = g.aki_search(start, matrix, gold_positions)
    elif player_type == 'Jocke':
        path = g.jocke_search(start, matrix, gold_positions)
    elif player_type == 'Micko':
        path = g.micko_search(start, matrix, gold_positions) 
    elif player_type == 'Uki':
        path = g.uki_search(start, matrix, gold_positions)
    else:
        path = []
        

    return JsonResponse({'path': path})
  

class Graph(object):
    def __init__(self, height, width):
        self.visitedS = []
        self.height = height
        self.width = width

    def get_moves(self, position):
        (x, y) = position
        moves = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  
        moves = [(x, y) for (x, y) in moves if 0 <= x < self.height and 0 <= y < self.width]
        return moves

    def make_matrix(self, height, width, tiles):
        matrix = []
        for i in range(height):
            matrix.append(tiles[i * width:i * width + width][:width]) 
        return matrix

    def reconstruct_path(self, came_from, start, end):
        path = []
        current = end
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    
    def aki_search(self, start, matrixCost, gold_positions):
        paths = [] 
        gold_positions = [(pos, i) for i, pos in enumerate(gold_positions)]  

        for _ in range(len(gold_positions)): 
            heap = [(0, start)]  
            visited = set([tuple(start)]) 
            came_from = {tuple(start): None} 

            while heap:
                
                cost, position = heapq.heappop(heap) 

                if list(position) in [pos for pos, _ in gold_positions]: 
                    path = self.reconstruct_path(came_from, start, position) 
                    if paths:
                        paths[-1].extend(path)  
                    else:
                        paths.append(path)
                    gold_positions.remove((list(position), [i for pos, i in gold_positions if pos == list(position)][0]))
                    start = position  

                    
                    heap = [(0, start)]
                    visited = set([tuple(start)])
                    came_from = {tuple(start): None}

                    break

                moves = self.get_moves(position)
                for move in moves: 
                    if tuple(move) not in visited and matrixCost[move[0]][move[1]] != 0: 
                        
                        heapq.heappush(heap, (cost + matrixCost[move[0]][move[1]], move))
                        visited.add(tuple(move))
                        came_from[tuple(move)] = position

        return paths


        




    

    from collections import deque

    def jocke(self, start, matrixCost, gold_positions):
        visited = set()
        queue = deque([(start, 0, [])])

        while queue:
            (x, y), cost, path = queue.popleft() 
            if (x, y) not in visited:
                visited.add((x, y))
                path = path + [(x, y)]

                if [x, y] in gold_positions:  
                    return path, (x, y)  

                for move in self.get_moves((x, y)):
                    if move not in visited:
                        new_cost = cost + matrixCost[move[0]][move[1]]
                        queue.append((move, new_cost, path))

        return None, None 

    def jocke_search(self, start, matrixCost, gold_positions):
        paths_to_gold = []
        while gold_positions:  
            path_to_gold, last_position = self.jocke(start, matrixCost, gold_positions)
            if path_to_gold:
                paths_to_gold.append(path_to_gold)
                gold_positions.remove(list(last_position))
                start = last_position  
        return paths_to_gold

    




    
    
    def get_moves(self, position):
        x, y = position
        moves = []

        possible_moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        for move in possible_moves:
            if 0 <= move[0] < self.height and 0 <= move[1] < self.width:
                moves.append(move)

        return moves
    
 

    import math

    def get_manhattan_heuristic(self, current_position, position, remaining_gold_positions):
        if not remaining_gold_positions:
            return 0

        distances = [abs(position[0] - pos[0]) + abs(position[1] - pos[1]) for pos in remaining_gold_positions]

        min_distance = min(distances)

        return min_distance




    def reconstruct_path2(self, came_from, start, end):
        path = []
        current = end
        while current != start:
            path.append(list(current))
            current = came_from[tuple(current)]
        path.append(start)
        path.reverse()
        return path
    

    def uki_search(self, start, matrix_cost, gold_positions):
        paths = []
        gold_positions = [(pos, i) for i, pos in enumerate(gold_positions)]
        gold_positions.sort(key=lambda x: (x[1], -x[0][0], -x[0][1]))
        visited = set()
        came_from = {}

        while gold_positions:
            stack = [(0, 0, 0, start)]
            heapq.heapify(stack)

            while stack:
                cost, collected_gold, id_value, position = heapq.heappop(stack)

                if list(position) in [pos for pos, _ in gold_positions]:
                    path = self.reconstruct_path2(came_from, start, position)
                    if paths:
                        paths[-1].extend(path)
                    else:
                        paths.append(path)
                    current_gold_position = next(i for i, pos in enumerate(gold_positions) if pos[0] == list(position))
                    gold_positions.pop(current_gold_position)
                    start = position
                    came_from = {start: None}
                    visited = {start}
                    break

                moves = self.get_moves(position)
                for move in moves:
                    if tuple(move) not in visited and matrix_cost[move[0]][move[1]] != 0:
                        heapq.heappush(stack, (cost + matrix_cost[move[0]][move[1]], -collected_gold-1, -id_value, move))
                        visited.add(tuple(move))
                        came_from[tuple(move)] = position

            paths = [path for path in paths if path]
            paths.sort(key=lambda x: (len(x), -x[-1][1] if len(x[-1]) > 1 else 0, x[-1][2] if len(x[-1]) > 2 else 0))

        return paths



    def micko_search(self, start, matrix_cost, gold_positions):
        paths = []
        gold_positions = [(pos, i) for i, pos in enumerate(gold_positions)]
        gold_positions.sort(key=lambda x: (x[1], -x[0][0], -x[0][1]))

        while gold_positions:
            heap = [(0, 0, 0, start)] 
            visited = set([tuple(start)])
            came_from = {tuple(start): None}

            while heap:
                
                cost, neg_gold_count, id_value, position = heapq.heappop(heap)

                heuristic_value = self.get_manhattan_heuristic(start, position, [pos for pos, _ in gold_positions])


                if list(position) in [pos for pos, _ in gold_positions]:
                    path = self.reconstruct_path2(came_from, start, position)
                    if paths:
                        paths[-1].extend(path[1:])  
                    else:
                        paths.append(path)
                    current_gold_position = next(i for i, pos in enumerate(gold_positions) if pos[0] == list(position))
                    gold_positions.pop(current_gold_position)
                    start = position  
                    visited.add(tuple(position))  
                    break

                moves = self.get_moves(position)
                for move in moves:
                    if tuple(move) not in visited and matrix_cost[move[0]][move[1]] != 0:
                        heapq.heappush(heap, (cost + matrix_cost[move[0]][move[1]], -neg_gold_count-1, -heuristic_value, move))
                        visited.add(tuple(move))
                        came_from[tuple(move)] = position

        return paths


    

    