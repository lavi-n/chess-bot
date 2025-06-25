import chess
import random
import math
import numpy as np
import os
from tensorflow import keras
from keras.models import load_model

TOTAL_POSSIBLE_MOVES_SIZE = 4128

def board_to_tensor(board):
    """
    Converts a chess.Board object into a numerical tensor for a neural network.
    This tensor represents the board state using 20 feature planes (8x8),
    including piece positions for both colors, turn, castling rights,
    en passant square, and move clocks. This format is suitable for
    input into a convolutional neural network.
    """
    num_features = 20
    tensor = np.zeros((8, 8, num_features), dtype=np.float32)
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel_offset = 6 if piece.color == chess.BLACK else 0
            tensor[rank, file, piece_to_channel[piece.piece_type] + channel_offset] = 1
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 13] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 14] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 15] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 16] = 1
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[rank, file, 17] = 1
    tensor[:, :, 18] = board.halfmove_clock / 100.0
    tensor[:, :, 19] = board.fullmove_number / 150.0
    return tensor

def map_move_to_index(move):
    """
    Maps a chess.Move object to a unique integer index.
    Regular moves are mapped based on their from and to squares.
    Promotion moves are mapped to a separate block of indices,
    distinguishing between different promotion pieces. This creates
    a flat representation for all possible moves, which is used
    as the output policy of the neural network.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    base_index = from_sq * 64 + to_sq
    if move.promotion is not None:
        promotion_square_idx = chess.square_file(to_sq)
        if move.promotion == chess.QUEEN: promotion_offset = 0
        elif move.promotion == chess.ROOK: promotion_offset = 1 * 8
        elif move.promotion == chess.BISHOP: promotion_offset = 2 * 8
        elif move.promotion == chess.KNIGHT: promotion_offset = 3 * 8
        else: return None
        index = 4096 + promotion_offset + promotion_square_idx
        if index >= TOTAL_POSSIBLE_MOVES_SIZE: return None
        return index
    return base_index

def map_index_to_move(index, board):
    """
    Maps an integer index back to a chess.Move object given a board state.
    This function is the inverse of map_move_to_index. It decodes
    the index to find the from and to squares, and in the case of a
    promotion, the correct promotion piece. It then validates if the
    reconstructed move is legal on the given board.
    """
    if index < 0 or index >= TOTAL_POSSIBLE_MOVES_SIZE:
        return None
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        return chess.Move(from_sq, to_sq)
    else:
        promotion_part = index - 4096
        promotion_type_offset = promotion_part // 8
        promotion_square_idx = promotion_part % 8
        promotion_piece = None
        if promotion_type_offset == 0: promotion_piece = chess.QUEEN
        elif promotion_type_offset == 1: promotion_piece = chess.ROOK
        elif promotion_type_offset == 2: promotion_piece = chess.BISHOP
        elif promotion_type_offset == 3: promotion_piece = chess.KNIGHT
        target_to_square_rank = 7 if board.turn == chess.WHITE else 0
        target_to_square = chess.square(promotion_square_idx, target_to_square_rank)
        for move in board.legal_moves:
            if move.to_square == target_to_square and move.promotion == promotion_piece:
                return move
        return None

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, policy_prior=None):
        """
        Initializes a node in the Monte Carlo Tree Search.
        Each node stores the game state, its relationship to other nodes
        (parent and children), statistics for UCB calculation (visits, value),
        a list of unexplored moves, and the policy priors from the
        neural network for guiding the search.
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.unexplored_moves = list(game_state.legal_moves)
        self.player_to_move = game_state.turn
        self.policy_prior = policy_prior if policy_prior is not None else {}

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
    def __init__(self, game, model_path="chess_ai_model.h5", c_param=0.8):
        """
        Initializes the MCTS agent.
        This involves setting the game interface, the exploration parameter (c_param),
        and loading the pre-trained neural network model that will guide the
        search by providing policy and value estimates.
        """
        self.game = game
        self.c_param = c_param
        self.trained_model = self._load_model(model_path)
        self.root = None

    def _load_model(self, model_path):
        """
        Loads the pre-trained Keras model from the specified path.
        If the model file does not exist, it prints an error and exits,
        as the model is essential for the MCTS agent's operation.
        """
        if not os.path.exists(model_path):
            print(f"Error: Trained model not found at {model_path}.")
            exit()
        print(f"Loading trained model from {model_path}...")
        return load_model(model_path)

    def _predict(self, board_state):
        """
        Performs a forward pass through the neural network.
        It takes a board state, converts it to a tensor, and feeds it to the
        model to get a policy (move probabilities) and a value estimate
        (predicted game outcome). This is the core of the NN-guided search.
        """
        input_tensor = board_to_tensor(board_state)
        input_tensor_batch = np.expand_dims(input_tensor, axis=0)
        policy_probs, value_estimate = self.trained_model.predict(input_tensor_batch, verbose=0)
        return policy_probs[0], value_estimate[0][0]

    def choose_move(self, current_board_state, num_iterations):
        """
        Performs MCTS iterations to choose the best move.
        It initializes the search from the current state, using the NN to get
        initial policy priors. It then runs a specified number of simulations
        (select, expand, backpropagate). Finally, it selects the move that
        was visited most often, which is considered the most robust choice.
        """
        root_policy_probs, root_value_estimate = self._predict(current_board_state)
        legal_moves_prior = {}
        for move in self.game.legal_moves(current_board_state):
            move_idx = map_move_to_index(move)
            if move_idx is not None and move_idx < TOTAL_POSSIBLE_MOVES_SIZE:
                legal_moves_prior[move] = root_policy_probs[move_idx]
            else:
                legal_moves_prior[move] = 0.0
        sum_priors = sum(legal_moves_prior.values())
        if sum_priors > 0:
            legal_moves_prior = {move: prob / sum_priors for move, prob in legal_moves_prior.items()}
        else:
            num_legal_moves = len(self.game.legal_moves(current_board_state))
            if num_legal_moves > 0:
                legal_moves_prior = {move: 1.0 / num_legal_moves for move in self.game.legal_moves(current_board_state)}
            else:
                return None
        self.root = MCTSNode(current_board_state, policy_prior=legal_moves_prior)
        self._backpropagate(self.root, root_value_estimate)
        for i in range(num_iterations):
            node = self._select_node(self.root)
            winner = self.game.get_winner(node.game_state)
            if winner is None:
                new_child_node, value_estimate = self._expand_node(node)
                if new_child_node:
                    self._backpropagate(new_child_node, value_estimate)
                else:
                    final_winner = self.game.get_winner(node.game_state)
                    if final_winner is not None:
                        self._backpropagate(node, final_winner)
            else:
                self._backpropagate(node, winner)
        best_move = None
        most_visits = -1
        if not self.root.children:
            return None
        for move, child_node in self.root.children.items():
            if child_node.visits > most_visits:
                most_visits = child_node.visits
                best_move = move
        return best_move

    def _select_node(self, node):
        """
        Selects a child node to traverse using the PUCT formula.
        This function iteratively applies the PUCT (Polynomial Upper Confidence
        Trees) algorithm to navigate down the tree, balancing exploitation
        (choosing moves with high estimated value) and exploration (choosing
        less-visited moves, guided by the policy prior). The process stops
        when a leaf or terminal node is reached.
        """
        while not node.unexplored_moves and node.children:
            best_child = None
            best_ucb_score = -float('inf')
            current_player_at_node_turn = node.player_to_move
            for move, child_node in node.children.items():
                Q_value = child_node.value_sum / child_node.visits if child_node.visits > 0 else 0.0
                P_value = node.policy_prior.get(move, 0.0)
                exploration_term = self.c_param * P_value * (math.sqrt(node.visits) / (1 + child_node.visits))
                ucb_score = Q_value + exploration_term
                if child_node.visits == 0:
                    ucb_score = float('inf')
                if ucb_score > best_ucb_score:
                    best_ucb_score = ucb_score
                    best_child = child_node
            if best_child is None:
                legal_moves_at_node = self.game.legal_moves(node.game_state)
                if legal_moves_at_node:
                    unvisited_children_moves = [m for m in legal_moves_at_node if m not in node.children or node.children[m].visits == 0]
                    if unvisited_children_moves:
                        best_child = node.children.get(random.choice(unvisited_children_moves))
                    else:
                        best_child = random.choice(list(node.children.values()))
                if best_child is None:
                     break
            node = best_child
            if self.game.game_over_check(node.game_state):
                 break
        return node

    def _expand_node(self, node):
        """
        Expands the current node by creating a new child node.
        It selects an unexplored move (prioritizing those with higher policy
        priors), creates a new node for the resulting game state, and gets
        the policy/value estimate for this new node from the neural network.
        This adds new information to the search tree.
        """
        if not node.unexplored_moves:
            return None, None
        candidate_moves = [(move, node.policy_prior.get(move, 0.0)) for move in node.unexplored_moves]
        candidate_moves.sort(key=lambda x: x[1], reverse=True)
        if not candidate_moves:
            move_to_explore = random.choice(node.unexplored_moves)
        else:
            move_to_explore = candidate_moves[0][0]
        node.unexplored_moves.remove(move_to_explore)
        new_game_state = self.game.make_move(node.game_state, move_to_explore)
        child_policy_probs, child_value_estimate = self._predict(new_game_state)
        child_legal_moves_prior = {}
        legal_moves_for_child = self.game.legal_moves(new_game_state)
        sum_child_priors = 0.0
        for move in legal_moves_for_child:
            move_idx = map_move_to_index(move)
            if move_idx is not None and move_idx < TOTAL_POSSIBLE_MOVES_SIZE:
                child_policy_prior_value = child_policy_probs[move_idx]
                child_legal_moves_prior[move] = child_policy_prior_value
                sum_child_priors += child_policy_prior_value
            else:
                child_legal_moves_prior[move] = 0.0
        if sum_child_priors > 0:
            child_legal_moves_prior = {move: prob / sum_child_priors for move, prob in child_legal_moves_prior.items()}
        else:
            if legal_moves_for_child:
                child_legal_moves_prior = {move: 1.0 / len(legal_moves_for_child) for move in legal_moves_for_child}
        new_child_node = MCTSNode(new_game_state, parent=node, move=move_to_explore, policy_prior=child_legal_moves_prior)
        node.children[move_to_explore] = new_child_node
        return new_child_node, child_value_estimate

    def _backpropagate(self, node, value):
        """
        Backpropagates the value up the tree from a leaf node.
        The `value` (from a game outcome or NN estimate) is propagated
        upwards to the root. Each ancestor's visit count is incremented,
        and its value sum is updated. The value is negated at each step
        to reflect the alternating perspectives of the players.
        """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if current_node.player_to_move == chess.WHITE:
                current_node.value_sum += value
            else:
                current_node.value_sum += -value
            current_node = current_node.parent

if __name__ == "__main__":
    game = ChessGame()
    mcts_agent = MCTS(game, model_path="chess_ai_model.h5", c_param=0.8)

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
                game.display_board(game.board)
            else:
                print("MCTS could not find a move (game might be over or no legal moves).")
                break
        else:
            print("\nYour turn (Black).")
            while True:
                try:
                    human_move_str = input("Enter your move (e.g., 'e7e5'): ")
                    human_move = chess.Move.from_uci(human_move_str)
                    if human_move in game.legal_moves(game.board):
                        game.board.push(human_move)
                        game.display_board(game.board)
                        break
                    else:
                        print("Invalid move. Not legal. Try again.")
                except ValueError:
                    print("Invalid UCI format. Please use format 'e2e4'. Try again.")
        if game.game_over_check(game.board):
            print("\nGame Over!")
            winner_result = game.get_winner(game.board)
            if winner_result == 1:
                print("White wins!")
            elif winner_result == -1:
                print("Black wins!")
            else:
                print("It's a draw!")
            break
