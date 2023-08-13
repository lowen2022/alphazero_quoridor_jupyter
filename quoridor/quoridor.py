# Quoridor game implementation
import functools
import numpy as np
import itertools
import random
import heapq

class QuoridorState:
    '''
    Quoridor state container class - generalized by size N
    Attributes:
        positions: 2x2 tuple representing the positions of the players
        left_wall: 1x2 tuple representing the number of walls left for each player
        walls: NxNx2 numpy array representing the walls
    '''
    def __init__(self, N: int = 9, n_walls: int = 10, copy: 'QuoridorState' = None):
        self.N = N
        if copy is not None:
            self.N = copy.N
            self.positions = copy.positions.copy()
            self.left_wall = copy.left_wall.copy()
            self.walls = copy.walls.copy()
            self.board = copy.board.copy()
            self.move_cnt = copy.move_cnt
        else:
            self.positions = np.array([[0, self.N // 2], [self.N - 1, self.N // 2]])
            self.left_wall = np.array([n_walls, n_walls])
            self.walls = np.zeros((2, self.N - 1, self.N - 1), dtype=np.int8)
            self.board = self.init_board()
            self.move_cnt = 0


    def init_board(self):
        '''
        Returns a 2N-1x2N-1 numpy array representing the board
        '''
        board = np.zeros((self.N * 2 - 1, self.N * 2 - 1), dtype=np.int8)
        board[1::2, 1::2] = 1
        board[self.positions[0, 0] * 2, self.positions[0, 1] * 2] = 2
        board[self.positions[1, 0] * 2, self.positions[1, 1] * 2] = 3
        return board

    def copy(self):
        return QuoridorState(copy=self)
    
    def encode(self, player):
        '''
        Returns 4xNxN numpy array representing the state
        channel 1: position of current player
        channel 2: position of opponent player
        channel 3: horizontal walls
        channel 4: vertical walls
        If player is 1, the board is flipped vertically
        '''
        encoded = np.zeros((4, self.N, self.N), dtype=np.float32)
        
        encoded[player, self.positions[0, 0], self.positions[0, 1]] = 1
        encoded[1 - player, self.positions[1, 0], self.positions[1, 1]] = 1
        encoded[(0, 1), :, :] = encoded[(0, 1), :, :] if player == 0 else np.flip(encoded[(0, 1), :, :], axis=1)

        walls_1 = self.walls[0, :, :] if player == 0 else np.flipud(self.walls[1, :, :])
        walls_2 = self.walls[1, :, :] if player == 0 else np.flipud(self.walls[0, :, :])
        encoded[2, :, :] = np.pad(walls_1 == 1, ((0, 1), (0, 1)), 'constant', constant_values=0)
        encoded[3, :, :] = np.pad(walls_2 == 1, ((0, 1), (0, 1)), 'constant', constant_values=0)
        return encoded

class Quoridor:
    '''
    Quoridor rule management class
    '''
    def __init__(self, N, n_walls):
        self.N = N
        self.n_walls = n_walls
    
    def get_initial_state(self):
        '''
        Returns the initial state of the game
        '''
        state = QuoridorState(self.N, self.n_walls)
        return state
    
    def _search_on_board(self, state: QuoridorState, player):
        '''
        Returns the distance of shortest path to the goal of given player using a* algorithm.
        1 is wall, 0 is path
        if player 0: end at (2N - 2, *)
        if player 1: end at (0, *)
        heuristic: column-distance to the goal
        '''
        board = state.board
        now_pos = state.positions[player] * 2
        queue = []
        heapq.heappush(queue, (2 * self.N - 2, 0, 0, now_pos))
        
        visited = np.zeros((2 * self.N - 1, 2 * self.N - 1), dtype=np.int8)
        temp = 0
        while queue:
            _, _, g, pos = heapq.heappop(queue)
            if pos[0] < 0 or pos[0] > 2 * self.N - 2 or pos[1] < 0 or pos[1] > 2 * self.N - 2:
                continue
            if board[*pos] == 1 or visited[*pos] == 1:
                continue
            if pos[0] == (2 * self.N - 2) * (1 - player):
                return g
            
            visited[*pos] = 1
            for i in range(4):
                e = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])[i]
                new_pos = pos + e
                h = (2 * self.N - 2 - new_pos[0]) * (1 - player) + (2 * new_pos[0]) * player
                heapq.heappush(queue, (h + g + 1, temp := temp+1, g + 1, pos + e))
        
        return -1


    def is_valid_wall(self, state: QuoridorState):
        '''
        Returns True if the state is valid, False otherwise
        Conditions for a valid state:
            - Walls are not blocking the path to the goal
        '''
        return self._search_on_board(state, 0) != -1 and self._search_on_board(state, 1) != -1
    
    def _search_valid_moves(self, state: QuoridorState, player):
        '''
        Returns a list of valid moves from the given position using dfs
        1 is wall, 0 is path
        2 is player 0, 3 is player 1
        stack: [(pos, n_step)]
        n_step stops at 2
        reset step when board[pos] == 2 or 3
        '''
        board = state.board
        now_pos = state.positions[player] * 2
        movable = []
        stack = []
        stack.append((now_pos, 0))
        visited = np.zeros((2 * self.N - 1, 2 * self.N - 1), dtype=np.int8)
        while len(stack) > 0:
            pos, step = stack.pop()
            if pos[0] < 0 or pos[0] > 2 * self.N - 2 or pos[1] < 0 or pos[1] > 2 * self.N - 2:
                continue
            if board[*pos] == 1 or visited[*pos] == 1:
                continue
            if board[*pos] == 2 or board[*pos] == 3:
                step = 0
            visited[*pos] = 1
            if step == 2:
                movable.append(pos // 2)
                continue
            for e in np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]):
                stack.append((pos + e, step + 1))

        return movable


    def is_valid_move(self, state: QuoridorState, next_pos, player):
        '''
        Returns True if the next state is a valid move, False otherwise
        Conditions for a valid move:
            - The player is moving to a valid position
        '''
        cvt_pos = np.array(next_pos)
        if any([np.array_equal(cvt_pos, val_mov) for val_mov in self._search_valid_moves(state, player)]):
            return True
        else:
            return False


    def get_next_state(self, state: QuoridorState, action: tuple, player: int):
        '''
        Returns the next state of the game given the current state and action
        '''
        action_type, row, col = action
        
        next_state = QuoridorState(copy=state)
        next_state.move_cnt += 1

        if action_type == 0:
            if self.is_valid_move(next_state, (row, col), player):
                next_state.board[*next_state.positions[player] * 2] = 0
                next_state.board[*np.array((row, col)) * 2] = player + 2
                next_state.positions[player] = (row, col)
                return next_state
            else:
                # print('Invalid move')
                return None
        else:
            if next_state.left_wall[player] == 0:
                # print('No wall left')
                return None
            
            hv = action_type - 1
            if next_state.walls[hv, row, col] != 0:
                # print('Invalid wall')
                return None
            
            next_state.walls[hv, row, col] = 1
            next_state.walls[1 - hv, row, col] = -1
            if hv == 0 and col > 0:
                next_state.walls[0, row, col - 1] = -1
            if hv == 1 and row > 0:
                next_state.walls[1, row - 1, col] = -1
            if hv == 0 and col < self.N - 2:
                next_state.walls[0, row, col + 1] = -1
            if hv == 1 and row < self.N - 2:
                next_state.walls[1, row + 1, col] = -1
            next_state.board[
                row * 2 - hv + 1 : row * 2 + hv + 2,
                col * 2 - (1 - hv) + 1 : col * 2 + (1 - hv) + 2
            ] = 1
            next_state.left_wall[player] -= 1
            if not self.is_valid_wall(next_state):
                # print('Invalid wall')
                return None

            return next_state

    def get_valid_actions(self, state: QuoridorState, player: int):
        moves = self._search_valid_moves(state, player)
        walls = [
            (hv, r, c)
            for hv in range(2) for r in range(self.N - 1) for c in range(self.N - 1) 
            if self.get_next_state(state, (hv + 1, r, c), player) is not None
        ]
        actions = np.zeros((3, self.N, self.N))
        for move in moves:
            actions[0, *move] = 1
        for hv, r, c in walls:
            actions[1 + hv, r, c] = 1
        return actions

    def check_win(self, state: QuoridorState, player):
        '''
        Returns True if the player wins, False otherwise
        '''
        if state.positions[player][0] == (self.N - 1) * (1 - player):
            return True
        else:
            return False
    
    def get_draw_value(self, state: QuoridorState, player: int):
        '''
        Returns the reward of the given state. Possibly value can be heuristic, not only win-lose.
        '''
        p_value = self._search_on_board(state, player)
        o_value = self._search_on_board(state, 1 - player)

        if p_value + o_value == 0:
            print(p_value, o_value)
            print(state.board)
            print(state.positions)
        return p_value / (p_value + o_value)
        
    def get_value_and_terminated(self, state: QuoridorState, player: int):
        '''
        Returns whether the game is terminated and the reward of the given state.
        If the game progresses more than 50 moves, the game is forced to terminate.
        '''
        if state.move_cnt > 50:
            return True, True, self.get_draw_value(state, player)
        if self.check_win(state, player):
            return True, False, 1
        else:
            return False, False, 0