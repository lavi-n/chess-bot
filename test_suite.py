import chess
import time

from chessboardMCTS import MCTS as MCTS_NN_Agent, ChessGame as Game_MCTS
from chessboardMCTSrandom import MCTS as MCTS_Random_Agent, ChessGame as Game_MCTS_Random
from chessboardrandom import RandomChessAgent, ChessGame as Game_Random

def run_single_game(white_agent, black_agent, game_instance, white_agent_name, black_agent_name):
    """
    Runs a single, automated game between two provided agents.
    It handles making moves for each agent in turn until the game
    is over, returning 1 for a white win, -1 for a black win,
    and 0 for a draw.
    """
    board = game_instance.board
    board.reset()

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            move = white_agent.choose_move(board.copy())
        else:
            move = black_agent.choose_move(board.copy())

        if move in board.legal_moves:
            board.push(move)
        else:
            return -1 if board.turn == chess.WHITE else 1

    outcome = board.outcome(claim_draw=True)
    if outcome.winner == chess.WHITE:
        return 1
    elif outcome.winner == chess.BLACK:
        return -1
    else:
        return 0

def run_test_suite(num_games):
    """
    Organizes and runs a tournament between different chess agents.
    This function initializes the agents, sets up matchups, and runs
    a specified number of games for each pair. It tracks the wins,
    losses, and draws, and prints a comprehensive summary of the
    results at the end, allowing for performance comparison between
    the different AI implementations.
    """
    game_for_nn = Game_MCTS()
    game_for_mcts_random = Game_MCTS_Random()
    game_for_random = Game_Random()

    agent_nn_mcts = MCTS_NN_Agent(game_for_nn, model_path="chess_ai_model.h5")
    agent_mcts_random = MCTS_Random_Agent(game_for_mcts_random)
    agent_random = RandomChessAgent(game_for_random)

    agents = {
        "NN_MCTS": agent_nn_mcts,
        "MCTS_Random": agent_mcts_random,
        "Random": agent_random
    }

    matchups = [
        ("NN_MCTS", "MCTS_Random"),
        ("NN_MCTS", "Random"),
        ("MCTS_Random", "Random")
    ]

    results = {}

    print("--- Starting Chess AI Test Suite ---")
    print(f"Each matchup will be played {num_games} times.")

    for agent1_name, agent2_name in matchups:
        print(f"\n--- Matchup: {agent1_name} (White) vs. {agent2_name} (Black) ---")
        
        scores = {"White": 0, "Black": 0, "Draw": 0}
        start_time = time.time()

        for i in range(num_games):
            game_result = run_single_game(agents[agent1_name], agents[agent2_name], game_for_nn, agent1_name, agent2_name)
            if game_result == 1:
                scores["White"] += 1
            elif game_result == -1:
                scores["Black"] += 1
            else:
                scores["Draw"] += 1
            
            print(f"  Game {i+1}/{num_games} completed. Current Score: {agent1_name}: {scores['White']}, {agent2_name}: {scores['Black']}, Draws: {scores['Draw']}", end='\r')

        end_time = time.time()
        print("\n" + "="*50)
        print(f"  Final Score ({agent1_name} vs {agent2_name}):")
        print(f"    {agent1_name} (White) Wins: {scores['White']}")
        print(f"    {agent2_name} (Black) Wins: {scores['Black']}")
        print(f"    Draws: {scores['Draw']}")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        print("="*50)
        
        results[f"{agent1_name}_vs_{agent2_name}"] = scores


    print("\n--- Test Suite Summary ---")
    for matchup, score in results.items():
        print(f"Matchup: {matchup}")
        agent1, agent2 = matchup.split('_vs_')
        print(f"  {agent1} Wins: {score['White']}")
        print(f"  {agent2} Wins: {score['Black']}")
        print(f"  Draws: {score['Draw']}")
        print("-" * 20)

if __name__ == "__main__":
    run_test_suite(num_games=1000)
