import chess
import random
import math

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        """
        Initializes a node in the Monte Carlo Tree Search.
        Each node stores the game state, its relationship to other nodes
        (parent and children), statistics for UCB calculation (visits, wins),
        and a list of unexplored moves from the current state.
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.unexplored_moves = list(game_state.legal_moves)
        self.player_to_move = game_state.turn

class ChessGame:
    def __init__(self):
        """Initializes a new chess game with a standard starting board."""
        self.board = chess.Board()

    def get_game_state(self):
        """Returns a copy of the current board state."""
        return self.board.copy()

    def legal_moves(self, board_state):
        """Returns a list of legal moves for the given board state."""
        return list(board_state.legal_moves)

    def make_move(self, board_state, move):
        """Applies a move to a board state and returns the new state."""
        new_board = board_state.copy()
        new_board.push(move)
        return new_board

    def game_over_check(self, board_state):
        """Checks if the game is over for the given board state."""
        return board_state.is_game_over()

    def get_winner(self, board_state):
        """
        Determines the winner of the game.
        Returns 1 for White win, -1 for Black win, 0 for draw,
        or None if the game is not over.
        """
        outcome = board_state.outcome()
        if outcome is None:
            return None
        if outcome.winner == chess.WHITE:
            return 1
        elif outcome.winner == chess.BLACK:
            return -1
        else:
            return 0

    def get_current_player(self, board_state):
        """Returns the color of the player to move (chess.WHITE or chess.BLACK)."""
        return board_state.turn

    def display_board(self, board_state):
        """Prints the ASCII representation of the board."""
        print(board_state)
        print(f"To move: {'White' if board_state.turn == chess.WHITE else 'Black'}")
        print("-" * 30)

class MCTS:
    def __init__(self, game, c_param=1.4):
        """
        Initializes the MCTS agent.
        This sets the game interface and the exploration parameter (c_param)
        used in the UCB1 formula to balance exploration and exploitation
        during the tree search.
        """
        self.game = game
        self.c_param = c_param
        self.root = None

    def choose_move(self, current_board_state, num_iterations):
        """
        Performs MCTS iterations to choose the best move.
        This is the main entry point for the agent. It runs a specified number
        of simulations (select, expand, simulate, backpropagate). After the
        simulations, it selects the move leading to the child node with the
        highest win rate as the best move to play.
        """
        self.root = MCTSNode(current_board_state)
        for _ in range(num_iterations):
            node = self._select_node(self.root)
            winner = self.game.get_winner(node.game_state)
            if winner is None:
                node = self._expand_node(node)
                if node:
                    simulation_result = self._simulate_game(node.game_state)
                    self._backpropagate(node, simulation_result)
            else:
                self._backpropagate(node, winner)
        best_move = None
        best_win_rate = -1
        most_visits = -1
        for move, child_node in self.root.children.items():
            if child_node.visits == 0:
                continue
            current_win_rate = child_node.wins / child_node.visits
            if current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                best_move = move
                most_visits = child_node.visits
            elif current_win_rate == best_win_rate:
                if child_node.visits > most_visits:
                    best_move = move
                    most_visits = child_node.visits
        return best_move

    def _select_node(self, node):
        """
        Selects a child node to traverse using the UCB1 formula.
        This function repeatedly applies the UCB1 formula to navigate down
        the tree, balancing exploitation (choosing moves with high win rates)
        and exploration (choosing less-visited moves). The process stops
        when an un-expanded (leaf) node is reached.
        """
        while node.unexplored_moves == [] and node.children:
            best_child = None
            best_ucb1 = -float('inf')
            for move, child_node in node.children.items():
                if child_node.visits == 0:
                    ucb1 = float('inf')
                else:
                    value_term = - (child_node.wins / child_node.visits)
                    exploration_term = self.c_param * math.sqrt(math.log(node.visits) / child_node.visits)
                    ucb1 = value_term + exploration_term
                if ucb1 > best_ucb1:
                    best_ucb1 = ucb1
                    best_child = child_node
            node = best_child
            if node is None:
                break
            winner = self.game.get_winner(node.game_state)
            if winner is not None:
                return node
        return node

    def _expand_node(self, node):
        """
        Expands the current node by creating one new child node.
        It randomly selects one of the unexplored legal moves from the
        current node's state, creates a new child node for that move,
        and adds it to the tree. This new node is then the subject
        of the simulation phase.
        """
        if not node.unexplored_moves:
            return None
        move_to_explore = random.choice(node.unexplored_moves)
        node.unexplored_moves.remove(move_to_explore)
        new_game_state = self.game.make_move(node.game_state, move_to_explore)
        new_child_node = MCTSNode(new_game_state, parent=node, move=move_to_explore)
        node.children[move_to_explore] = new_child_node
        return new_child_node

    def _simulate_game(self, game_state):
        """
        Simulates a random playout from a given game state.
        From the provided state, this function plays random legal moves
        for both sides until the game ends (win, loss, or draw). It then
        returns the outcome of this "rollout" from the perspective of the
        player who was to move at the start of the simulation.
        """
        current_rollout_board = game_state.copy()
        original_player_to_move = current_rollout_board.turn
        while not self.game.game_over_check(current_rollout_board):
            legal_moves = self.game.legal_moves(current_rollout_board)
            if not legal_moves:
                break
            random_move = random.choice(legal_moves)
            current_rollout_board.push(random_move)
        winner_raw = self.game.get_winner(current_rollout_board)
        if winner_raw is None:
            return 0
        if original_player_to_move == chess.WHITE:
            return winner_raw
        else:
            return -winner_raw

    def _backpropagate(self, node, simulation_result):
        """
        Backpropagates the simulation result up the tree.
        The result of a simulation is propagated from the expanded node
        up to the root. Each ancestor's visit count is incremented, and
        its win count is updated based on the result, adjusted for the
        player's perspective at that node.
        """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            sim_starter_player_color = node.player_to_move
            if current_node.player_to_move == sim_starter_player_color:
                current_node.wins += simulation_result
            else:
                current_node.wins += -simulation_result
            current_node = current_node.parent

if __name__ == "__main__":
    game = ChessGame()
    mcts_agent = MCTS(game, c_param=1.4)
    print("Initial Board:")
    game.display_board(game.board)
    while not game.game_over_check(game.board):
        current_player = game.get_current_player(game.board)
        if current_player == chess.WHITE:
            print("\nMCTS (White) is thinking...")
            best_move = mcts_agent.choose_move(game.board.copy(), num_iterations=1000)
            if best_move:
                print(f"MCTS (White) chooses: {best_move.uci()}")
                game.board.push(best_move)
            else:
                print("MCTS could not find a move.")
                break
        else:
            print("\nYour turn (Black).")
            while True:
                try:
                    human_move_str = input("Enter your move (e.g., 'e7e5'): ")
                    human_move = chess.Move.from_uci(human_move_str)
                    if human_move in game.legal_moves(game.board):
                        game.board.push(human_move)
                        break
                    else:
                        print("Invalid move. Not legal. Try again.")
                except ValueError:
                    print("Invalid UCI format. Please use format 'e2e4'. Try again.")
        game.display_board(game.board)
    print("\nGame Over!")
    winner_result = game.get_winner(game.board)
    if winner_result == 1:
        print("White wins!")
    elif winner_result == -1:
        print("Black wins!")
    else:
        print("It's a draw!")
