import chess
import chess.pgn
import os
import json

PGN_FILE_PATH = "filtered_games_final2.pgn"

def parse_pgn(pgn_file_path):
    """
    Parses a PGN file and extracts game data.
    Assumes the PGN file is already filtered.
    """
    parsed_games_data = []
    print(f"Opening PGN file for processing: {pgn_file_path}")
    game_count = 0
    try:
        with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
            print("Starting to parse games...")
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                game_count += 1
                headers = game.headers
                moves_uci = [move.uci() for move in game.mainline_moves()]
                if moves_uci:
                    time_control_str = headers.get("TimeControl", "0")
                    base_time_sec = 0
                    try:
                        if '+' in time_control_str:
                            base_time_sec = int(time_control_str.split('+')[0])
                        elif time_control_str.isdigit():
                            base_time_sec = int(time_control_str)
                    except ValueError:
                        pass
                    parsed_games_data.append({
                        "moves": moves_uci,
                        "result": headers.get("Result"),
                        "white_elo": int(headers.get("WhiteElo", 0)),
                        "black_elo": int(headers.get("BlackElo", 0)),
                        "time_control_base_sec": base_time_sec
                    })
                    print(f"Games parsed: {game_count}", end='\r')
        print(f"\nFinished parsing. Total games parsed: {game_count}")
    except FileNotFoundError:
        print(f"Error: PGN file not found at {pgn_file_path}")
    except Exception as e:
        print(f"An error occurred during PGN parsing: {e}")
    return parsed_games_data

if __name__ == "__main__":
    print("Starting PGN parsing script...")
    if not os.path.exists(PGN_FILE_PATH):
        print(f"Error: Filtered PGN file not found at {PGN_FILE_PATH}. Please ensure '{PGN_FILE_PATH}' exists in the same directory.")
        exit()

    parsed_games = parse_pgn(PGN_FILE_PATH)

    if parsed_games:
        output_file = "parsed_games_data.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for game_data in parsed_games:
                f.write(json.dumps(game_data) + '\n')
        print(f"Parsed game data saved to {output_file}")
    else:
        print("No games were parsed. Output file will not be created.")
