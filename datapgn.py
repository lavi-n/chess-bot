import chess.pgn
import os # Added import for os module

PGN_FILE_PATH = "filtered_games_final.pgn"
OUTPUT_PGN_FILE_PATH = "filtered_games_final2.pgn"
MIN_ELO_FILTER = 2500
ALLOWED_RESULTS = ["1-0", "0-1"]
MIN_TIME_CONTROL_BASE_SEC = 300

def parse_and_filter_pgn(pgn_file_path, output_pgn_file_path, min_elo, allowed_results, min_time_control_base_sec):
    """
    Parses a PGN file, filters games, and saves the results.
    It reads games one by one from a large PGN file. Each game is checked
    against criteria like player ELO ratings, game result (e.g., no draws),
    and time control. Games that meet all criteria are written to a new
    output PGN file for further processing.

    The output file is opened in 'w' (write) mode, meaning it will be
    overwritten if it already exists.
    """
    print(f"Attempting to process PGN file: {pgn_file_path}")
    game_count = 0
    filtered_count = 0

    # Ensure the output file is fresh by removing it if it exists
    if os.path.exists(output_pgn_file_path):
        try:
            os.remove(output_pgn_file_path)
            print(f"Removed existing output file: {output_pgn_file_path}")
        except OSError as e:
            print(f"Error removing existing output file {output_pgn_file_path}: {e}")
            print("Please ensure the file is not open by another program and you have write permissions.")
            return # Exit if we can't clean up the old file

    try:
        # Open in 'w' mode to overwrite/create the file
        with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file, \
             open(output_pgn_file_path, 'w', encoding='utf-8') as output_file: # Changed 'a' to 'w' as requested for overwriting
            print("Starting to parse and filter games...")
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                game_count += 1
                headers = game.headers
                result = headers.get("Result")
                
                # Filter by result
                if result not in allowed_results:
                    continue

                # Filter by ELO
                try:
                    white_elo = int(headers.get("WhiteElo", 0))
                    black_elo = int(headers.get("BlackElo", 0))
                    if white_elo < min_elo or black_elo < min_elo:
                        continue
                except ValueError:
                    # Skip games with non-integer ELO ratings
                    print(f"Warning: Skipping game {game_count} due to non-integer ELO rating.")
                    continue
                
                # Filter by TimeControl
                time_control_str = headers.get("TimeControl", "")
                base_time_sec = 0
                try:
                    if '+' in time_control_str:
                        base_time_sec = int(time_control_str.split('+')[0])
                    elif time_control_str.isdigit():
                        base_time_sec = int(time_control_str)
                except ValueError:
                    # If TimeControl is malformed, treat base_time_sec as 0 for filtering
                    pass 
                
                if base_time_sec < min_time_control_base_sec:
                    continue
                
                # If all filters pass, write the game to the output file
                output_file.write(str(game) + "\n\n")
                filtered_count += 1
                # Use carriage return for in-place update of count
                print(f"Filtered games: {filtered_count}, Game count: {game_count}", end='\r')
        
        print(f"\nFinished parsing. Total games found: {game_count}")
        print(f"Total games after filtering: {filtered_count}")

    except FileNotFoundError:
        print(f"\nError: Input PGN file not found at '{pgn_file_path}'.")
        print("Please ensure the PGN file exists in the same directory as the script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
        print("Please check your PGN file format and script permissions.")

if __name__ == "__main__":
    parse_and_filter_pgn(
        PGN_FILE_PATH,
        OUTPUT_PGN_FILE_PATH,
        MIN_ELO_FILTER,
        ALLOWED_RESULTS,
        MIN_TIME_CONTROL_BASE_SEC
    )
