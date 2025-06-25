# Chess AI Project

This repository contains Python scripts for building, training, and evaluating a Chess AI using Monte Carlo Tree Search (MCTS) with both random playouts and a neural network for policy and value predictions. It also includes scripts for processing chess game data (PGN files) to train the neural network.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Playing Against the AI](#playing-against-the-ai)
- [Testing and Evaluation](#testing-and-evaluation)
- [Agents Implemented](#agents-implemented)
- [Requirements](#requirements)

## Project Structure

-   `datapgn.py`: Filters large PGN (Portable Game Notation) chess game files based on criteria like ELO rating, game result, and time control.
-   `pgn_parser.py`: Parses the filtered PGN games and extracts move sequences, results, ELO ratings, and time controls into a structured format (JSON lines).
-   `model_trainer.py`: Defines, trains, and saves a neural network model. This model is used by the MCTS agent to predict move probabilities (policy) and game outcomes (value).
-   `chessboardMCTS.py`: Implements a Monte Carlo Tree Search (MCTS) agent that uses a pre-trained neural network for guided exploration and evaluation. It includes a basic game loop to play against this AI.
-   `chessboardMCTSrandom.py`: Implements a simpler MCTS agent that relies on random playouts for simulations, without using a neural network. It also includes a basic game loop.
-   `chessboardrandom.py`: Implements a basic chess game environment and a simple random move-choosing agent. It includes a human-vs-AI game loop where the AI makes random legal moves.
-   `test_suite.py`: A script to automate the evaluation of different AI agents (NN-guided MCTS, Random MCTS, Random Agent) against each other over multiple games.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install necessary Python packages:**
    ```bash
    pip install python-chess numpy tensorflow scikit-learn
    ```

3.  **Obtain PGN data:**
    Download a large PGN file containing chess games. You can find these on websites like Lichess or Chess.com. Place the downloaded PGN file in the project's root directory. For example, rename it to `games.pgn`.

## Data Preparation

Before training the neural network, you need to filter and parse the raw PGN data.

1.  **Filter PGN games:**
    -   Edit `datapgn.py` to specify your input PGN file (`PGN_FILE_PATH`) and desired output file (`OUTPUT_PGN_FILE_PATH`).
    -   Adjust filtering parameters (`MIN_ELO_FILTER`, `ALLOWED_RESULTS`, `MIN_TIME_CONTROL_BASE_SEC`) as needed.
    -   Run the script:
        ```bash
        python datapgn.py
        ```
    This will create `filtered_games_final2.pgn` (or whatever you set `OUTPUT_PGN_FILE_PATH` to) containing games that meet your criteria.

2.  **Parse filtered PGN games:**
    -   Ensure `PGN_FILE_PATH` in `pgn_parser.py` points to the output from `datapgn.py` (e.g., `filtered_games_final2.pgn`).
    -   Run the script:
        ```bash
        python pgn_parser.py
        ```
    This will generate `parsed_games_data.txt`, a file where each line is a JSON object representing a parsed game. This file is used for training.

## Model Training

Once the data is prepared, you can train the neural network.

1.  **Train the model:**
    -   Ensure `PARSED_GAMES_DATA_FILE` in `model_trainer.py` points to `parsed_games_data.txt`.
    -   Run the script:
        ```bash
        python model_trainer.py
        ```
    This script will define a convolutional neural network, load the parsed game data, split it into training and validation sets, and then train the model. The trained model (`chess_ai_model.h5`) will be saved in the project root.

    *Note: Training can take a significant amount of time depending on the dataset size and your hardware.*

## Playing Against the AI

You can play against the trained NN-guided MCTS agent or the random MCTS agent.

-   **Play against NN-guided MCTS:**
    Run `chessboardMCTS.py`. The AI (White) will make its move, and you (Black) will be prompted to enter your move in UCI format (e.g., `e7e5`).
    ```bash
    python chessboardMCTS.py
    ```

-   **Play against Random MCTS:**
    Run `chessboardMCTSrandom.py`. Similar to the above, you'll play against an MCTS agent using random playouts.
    ```bash
    python chessboardMCTSrandom.py
    ```

-   **Play against a Random Agent:**
    Run `chessboardrandom.py`. This is the simplest AI, making purely random legal moves.
    ```bash
    python chessboardrandom.py
    ```

## Testing and Evaluation

The `test_suite.py` script allows you to evaluate the performance of the different AI agents against each other.

1.  **Run the test suite:**
    -   You can adjust the `num_games` variable in `test_suite.py` to control how many games are played in each matchup.
    -   Run the script:
        ```bash
        python test_suite.py
        ```
    This will run games between NN_MCTS vs MCTS_Random, NN_MCTS vs Random, and MCTS_Random vs Random, printing the results to the console.

## Agents Implemented

-   **NN-guided MCTS (`chessboardMCTS.py`):** A sophisticated MCTS implementation that leverages a neural network to provide policy (move probabilities) and value (game outcome prediction) estimates. This guides the tree search more efficiently than pure random playouts.
-   **Random MCTS (`chessboardMCTSrandom.py`):** A standard MCTS implementation that uses purely random playouts to simulate games and evaluate nodes. It's computationally more intensive than NN-guided MCTS for achieving similar performance.
-   **Random Agent (`chessboardrandom.py`):** A baseline AI that simply chooses a random legal move at each turn.

## Requirements

-   Python 3.x
-   `python-chess` library
-   `numpy`
-   `tensorflow` (for `model_trainer.py` and `chessboardMCTS.py`)
-   `scikit-learn` (for `model_trainer.py` for data splitting)

Make sure all dependencies are installed using pip.
