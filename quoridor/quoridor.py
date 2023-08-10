import numpy as np
import itertools
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
        else:
            self.positions = np.array([[0, self.N // 2], [self.N - 1, self.N // 2]])
            self.left_wall = np.array([n_walls, n_walls])
            self.walls = np.zeros((2, self.N - 1, self.N - 1), dtype=np.int8)
            self.board = self.init_board()


    def init_board(self):
        '''
        Returns a 2N-1x2N-1 numpy array representing the board
        '''
        board = np.zeros((self.N * 2 - 1, self.N * 2 - 1), dtype=np.int8)
        board[1::2, 1::2] = 1
        board[self.positions[0, 0] * 2, self.positions[0, 1] * 2] = 2
        board[self.positions[1, 0] * 2, self.positions[1, 1] * 2] = 3
        return board


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
        Search path to the goal of given player using a* algorithm
        1 is wall, 0 is path
        if player 0: end at (2N, *)
        if player 1: end at (0, *)
        heuristic: manhattan distance to the goal
        '''
        board = state.board
        now_pos = state.positions[player] * 2
        queue = []
        heuristic = lambda pos: (2 * self.N - 2 - pos[0]) * (1 - player) + (2 * pos[0]) * player
        heapq.heappush(queue, (-heuristic(now_pos), -1, now_pos))
        visited = np.zeros((2 * self.N - 1, 2 * self.N - 1), dtype=np.int8)
        for cnt in itertools.count():
            if len(queue) == 0:
                return False
            _, __, pos = queue.pop()
            if pos[0] < 0 or pos[0] > 2 * self.N - 2 or pos[1] < 0 or pos[1] > 2 * self.N - 2:
                continue
            if board[*pos] == 1 or visited[*pos] == 1:
                continue
            if pos[0] == (2 * self.N - 2) * (1 - player):
                return True
            visited[*pos] = 1
            for i in range(4):
                e = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])[i]
                heapq.heappush(queue, (-heuristic(pos + e), 4 * cnt + i, pos + e))


    def is_valid_wall(self, state: QuoridorState):
        '''
        Returns True if the state is valid, False otherwise
        Conditions for a valid state:
            - Walls are not blocking the path to the goal
        '''
        for i in range(2):
            if not self._search_on_board(state, i):
                return False
        return True
    
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
                movable.append(pos)
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
        cvt_pos = np.array(next_pos) * 2
        if any([np.array_equal(cvt_pos, val_mov) for val_mov in self._search_valid_moves(state, player)]):
            return True
        else:
            return False


    def get_next_state(self, state: QuoridorState, action: tuple, player: int):
        '''
        Returns the next state of the game given the current state and action
        '''
        action_type, action_value = action
        next_state = QuoridorState(copy=state)
        if action_type == 0:
            if self.is_valid_move(next_state, action_value, player):
                next_state.board[*next_state.positions[player] * 2] = 0
                next_state.board[*np.array(action_value) * 2] = player + 2
                next_state.positions[player] = action_value
                return next_state
            else:
                print('Invalid move')
                return state
        else:
            hv, row, col = action_value
            if next_state.walls[hv, row, col] == 0:
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
                if self.is_valid_wall(next_state):
                    return next_state
                else:
                    return state
            else:
                return state

    def check_win(self, state: QuoridorState, player):
        '''
        Returns True if the player wins, False otherwise
        '''
        if state.positions[player][0] == (2 * self.N - 2) * (1 - player):
            return True
        else:
            return False

def parse_cmd(cmd: str) -> tuple:
    s = cmd.split(' ')
    if s[0] == 'move':
        return 0, (int(s[1]), int(s[2]))
    elif s[0] == 'wall':
        return 1, (int(s[1]), int(s[2]), int(s[3]))
    else:
        raise ValueError('Invalid action type')
        
qm = Quoridor(3, 10)
q = qm.get_initial_state()
q = qm.get_next_state(q, (1, (0, 1, 1)), 0)
print(q.board, q.left_wall)
print(q.walls)
