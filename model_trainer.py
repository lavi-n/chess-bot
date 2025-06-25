import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import chess
import json
import os
from keras.callbacks import EarlyStopping

TOTAL_POSSIBLE_MOVES_SIZE = 4128
PARSED_GAMES_DATA_FILE = "parsed_games_data.txt"

def board_to_tensor(board):
    """
    Converts a chess.Board object into a numerical tensor suitable for a neural network.
    This implementation includes common features:
    - Piece positions (12 channels: 6 white pieces, 6 black pieces)
    - Player to move (1 channel)
    - Castling rights (4 channels)
    - En passant target square (1 channel)
    - Halfmove clock (1 channel, scaled)
    - Fullmove number (1 channel, scaled)

    Total channels: 12 + 1 + 4 + 1 + 1 + 1 = 20 channels.
    Output shape: (8, 8, 20)
    """
    num_features = 20
    tensor = np.zeros((8, 8, num_features), dtype=np.float32)

    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    # 1. Piece positions (Channels 0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel_offset = 6 if piece.color == chess.BLACK else 0
            tensor[rank, file, piece_to_channel[piece.piece_type] + channel_offset] = 1

    # 2. Player to move (Channel 12)
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1 # All 64 squares get 1 if White to move

    # 3. Castling rights (Channels 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 13] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 14] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 15] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 16] = 1

    # 4. En passant target square (Channel 17)
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[rank, file, 17] = 1

    # 5. Halfmove clock (Channel 18) - moves since last capture or pawn advance
    tensor[:, :, 18] = board.halfmove_clock / 100.0

    # 6. Fullmove number (Channel 19) - total move count
    tensor[:, :, 19] = board.fullmove_number / 150.0

    return tensor

def map_move_to_index(move):
    """
    Maps a chess.Move object to a unique integer index.
    This implementation maps (from_square, to_square) to a unique index (0 to 4095).
    For promotions, it uses a fixed offset based on the promotion piece type,
    adding 4 promotion types for each of the 8 possible promotion "to" squares (rank 8 for white, rank 1 for black).

    Total indices: (64 * 64) + (4 * 8) = 4096 + 32 = 4128.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    base_index = from_sq * 64 + to_sq

    if move.promotion is not None:
        promotion_square_idx = chess.square_file(to_sq)

        if move.promotion == chess.QUEEN:
            promotion_offset = 0
        elif move.promotion == chess.ROOK:
            promotion_offset = 1 * 8
        elif move.promotion == chess.BISHOP:
            promotion_offset = 2 * 8
        elif move.promotion == chess.KNIGHT:
            promotion_offset = 3 * 8
        else:
            return None

        index = 4096 + promotion_offset + promotion_square_idx

        if index >= TOTAL_POSSIBLE_MOVES_SIZE:
             print(f"Warning: Calculated promotion index {index} exceeds TOTAL_POSSIBLE_MOVES_SIZE {TOTAL_POSSIBLE_MOVES_SIZE}. Move: {move.uci()}")
             return None
        return index

    return base_index

def data_generator(parsed_games_data, total_moves_size):
    """
    A Python generator that yields training examples one by one,
    avoiding loading the entire dataset into memory.
    """
    for i, game_data in enumerate(parsed_games_data):
        board = chess.Board()
        moves = game_data["moves"]
        result = game_data["result"]

        game_value_white_perspective = 0.0
        if result == "1-0":
            game_value_white_perspective = 1.0
        elif result == "0-1":
            game_value_white_perspective = -1.0
        
        for j, move_uci in enumerate(moves):
            current_board_state = board.copy()
            
            try:
                actual_move = chess.Move.from_uci(move_uci)
            except ValueError:
                # If a move_uci string is invalid, chess.Move.from_uci will raise ValueError.
                # We'll print a warning and skip this move, advancing the board with a null move
                # to maintain the game state for subsequent moves in the sequence if possible.
                print(f"Warning: Invalid UCI move '{move_uci}' encountered in game {i}. Skipping move.")
                # Pushing a null move allows the game state to conceptually advance without making an invalid move.
                # This might not be ideal for all scenarios, depending on how strict you want the data
                # generation to be for malformed PGN entries.
                board.push(chess.Move.null()) 
                continue # Skip this specific training example

            policy_idx = map_move_to_index(actual_move)
            
            if policy_idx is None or policy_idx >= total_moves_size:
                # If the move cannot be mapped to an index (e.g., due to an unhandled promotion type
                # or a calculated index out of bounds, though the bounds check should prevent the latter).
                # The move still needs to be made on the board to advance game state.
                board.push(actual_move)
                continue

            input_tensor = board_to_tensor(current_board_state)
            
            policy_label = np.zeros(total_moves_size, dtype=np.float32)
            policy_label[policy_idx] = 1.0

            value_label = game_value_white_perspective if current_board_state.turn == chess.WHITE else -game_value_white_perspective
            
            yield input_tensor, (policy_label, np.array([value_label], dtype=np.float32))

            board.push(actual_move)
            if board.is_game_over():
                break

if __name__ == "__main__":
    print("Starting Chess AI Model Training Script...")

    # --- 1. Load Data ---
    parsed_games = []
    if not os.path.exists(PARSED_GAMES_DATA_FILE):
        print(f"Error: Parsed game data file not found at {PARSED_GAMES_DATA_FILE}.")
        print("Please run pgn_parser.py first to generate the data.")
        exit()
    
    print(f"\nLoading parsed games from {PARSED_GAMES_DATA_FILE}...")
    try:
        with open(PARSED_GAMES_DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parsed_games.append(json.loads(line.strip()))
        print(f"Successfully loaded {len(parsed_games)} games.")
    except Exception as e:
        print(f"An error occurred while loading parsed game data: {e}")
        exit()

    if not parsed_games:
        print("No games were loaded. Cannot generate training examples.")
        exit()

    # --- 2. Split game data ---
    print("\nSplitting game data into training and validation sets...")
    train_games, val_games = train_test_split(parsed_games, test_size=0.15, random_state=42)
    
    print(f"Training games: {len(train_games)}")
    print(f"Validation games: {len(val_games)}")

    # --- 3. Calculate total moves (examples) in each dataset to determine steps_per_epoch ---
    # This process iterates through the generators once to count the actual number of
    # examples that will be yielded. This is necessary to correctly set `steps_per_epoch`
    # and `validation_steps` for the Keras `fit` method when using generators.
    # Be aware that for extremely large datasets, this counting phase can be time-consuming.
    
    print("\nCalculating steps per epoch...")
    train_total_examples = 0
    
    # Create a temporary generator for counting to avoid consuming the main one
    for _ in data_generator(train_games, TOTAL_POSSIBLE_MOVES_SIZE):
        train_total_examples += 1
        print(f"Counted {train_total_examples} training examples...", end='\r')
    
    val_total_examples = 0
   # Create a temporary generator for counting
    for _ in data_generator(val_games, TOTAL_POSSIBLE_MOVES_SIZE):
        val_total_examples += 1
        print(f"Counted {val_total_examples} validation examples...", end='\r')

    BATCH_SIZE = 32
    train_steps_per_epoch = train_total_examples // BATCH_SIZE
    val_steps_per_epoch = val_total_examples // BATCH_SIZE

    # Ensure at least one step if there are examples, to avoid division by zero
    if train_steps_per_epoch == 0 and train_total_examples > 0:
        train_steps_per_epoch = 1
    if val_steps_per_epoch == 0 and val_total_examples > 0:
        val_steps_per_epoch = 1

    print(f"Total training examples to yield: {train_total_examples}")  
    print(f"Total validation examples to yield: {val_total_examples}") 
    print(f"Training steps per epoch: {train_steps_per_epoch} (using BATCH_SIZE={BATCH_SIZE})")
    print(f"Validation steps per epoch: {val_steps_per_epoch} (using BATCH_SIZE={BATCH_SIZE})")


    # --- 3. Re-Create tf.data.Dataset from generator (as they were consumed during counting) ---
    output_signature = (
        tf.TensorSpec(shape=(8, 8, 20), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(TOTAL_POSSIBLE_MOVES_SIZE,), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )
    
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_games, TOTAL_POSSIBLE_MOVES_SIZE),
        output_signature=output_signature
    ).repeat()
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(val_games, TOTAL_POSSIBLE_MOVES_SIZE),
        output_signature=output_signature
    ).repeat()

    # --- 4. Batch and Prefetch the datasets for performance ---
    train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- 5. Define Model Architecture ---
    print("\nDefining neural network architecture...")
    board_input = keras.Input(shape=(8, 8, 20), name="board_input")
    x = layers.Conv2D(filters=256, kernel_size=3, padding='same')(board_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    def residual_block(input_tensor, filters):
        y = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(input_tensor)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(y)
        y = layers.BatchNormalization()(y)
        return layers.Add()([y, input_tensor])

    for _ in range(5):
        x = residual_block(x, filters=256)
        x = layers.ReLU()(x)

    policy_head = layers.Conv2D(filters=32, kernel_size=1, activation='relu', name='policy_conv')(x)
    policy_head = layers.Flatten()(policy_head)
    policy_output = layers.Dense(TOTAL_POSSIBLE_MOVES_SIZE, activation='softmax', name='policy_output')(policy_head)

    value_head = layers.Conv2D(filters=1, kernel_size=1, activation='relu', name='value_conv')(x)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(256, activation='relu')(value_head)
    value_output = layers.Dense(1, activation='tanh', name='value_output')(value_head)

    model = keras.Model(inputs=board_input, outputs=[policy_output, value_output])

    # --- 6. Compile and Train the Model ---
    print("\nCompiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss={
            'policy_output': keras.losses.CategoricalCrossentropy(),
            'value_output': keras.losses.MeanSquaredError()
        },
        metrics={
            'policy_output': ['accuracy'],
            'value_output': ['mae']
        },
        # ** ADDED: Loss weights to give more importance to the value head's loss **
        loss_weights={
            'policy_output': 1.0,
            'value_output': 5.0 # Start with 5.0, can be adjusted (e.g., 10.0) if needed
        }
    )
    model.summary()
    print("\nStarting model training...")

    # Define EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_value_output_loss', 
        patience=5,                      
        min_delta=0.001,                 
        mode='min',                      
        restore_best_weights=True,       
        verbose=1                        
    )

    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        steps_per_epoch=train_steps_per_epoch,   
        validation_steps=val_steps_per_epoch,     
        callbacks=[early_stopping_callback] 
    )

    print("\nModel training finished.")
    model_save_path = "chess_ai_model.h5"
    model.save(model_save_path)
    print(f"Trained model saved to {model_save_path}")