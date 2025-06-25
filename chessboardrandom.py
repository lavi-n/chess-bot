import chess
import random

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

class RandomChessAgent:
    def __init__(self, game_instance):
        """
        Initializes the random agent.
        It takes a game instance to interact with the chess environment,
        allowing it to query for legal moves.
        """
        self.game = game_instance

    def choose_move(self, current_board_state):
        """
        Chooses a move randomly from the available legal moves.
        If there are no legal moves (e.g., in a checkmate or stalemate
        position), it returns None.
        """
        legal_moves = self.game.legal_moves(current_board_state)
        if not legal_moves:
            return None
        return random.choice(legal_moves)

if __name__ == "__main__":
    game = ChessGame()
    random_agent = RandomChessAgent(game)
    user_color = random.choice([chess.WHITE, chess.BLACK])
    agent_color = chess.BLACK if user_color == chess.WHITE else chess.WHITE
    print(f"You are playing as {'White' if user_color == chess.WHITE else 'Black'}.")
    print("Initial Board:")
    game.display_board(game.board)
    print("Starting the game...")
    while not game.game_over_check(game.board):
        current_player = game.get_current_player(game.board)
        if current_player == user_color:
            print("Your turn:")
            legal_moves = game.legal_moves(game.board)
            print("Legal moves:")
            print([move.uci() for move in legal_moves])
            user_move_input = input("Enter your move in UCI format (e.g., e2e4): ")
            try:
                user_move = chess.Move.from_uci(user_move_input)
                if user_move in legal_moves:
                    game.board.push(user_move)
                else:
                    print("Illegal move. Try again.")
                    continue
            except ValueError:
                print("Invalid input format. Try again.")
                continue
        else:
            print(f"{'White' if current_player == chess.WHITE else 'Black'} (Random Agent) is making a move...")
            best_move = random_agent.choose_move(game.board.copy())
            if best_move:
                print(f"Agent chooses: {best_move}")
                game.board.push(best_move)
            else:
                print("Agent could not find a move.")
                break
        game.display_board(game.board)
    print("\nGame Over!")
    winner = game.get_winner(game.board)
    if winner == 1:
        print("White wins!")
    elif winner == -1:
        print("Black wins!")
    else:
        print("It's a draw!")
    if (winner == 1 and user_color == chess.WHITE) or (winner == -1 and user_color == chess.BLACK):
        print("You win!")
    elif winner == 0:
        print("It's a draw!")
    else:
        print("You lose!")
