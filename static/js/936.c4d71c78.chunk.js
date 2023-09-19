"use strict";(self.webpackChunkportfolio=self.webpackChunkportfolio||[]).push([[936],{8936:function(e,n,a){a.r(n),a.d(n,{codeString:function(){return r},markdownContent:function(){return o}});var o='\n# Connect Four AI - Minimax Algorithm with Alpha Beta Pruning\n\nConnect Four is a two-player connection game in which the players first choose a color and then take turns dropping colored discs from the top into a seven-column, six-row vertically suspended grid. The pieces fall straight down, occupying the lowest available space within the column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one\'s own discs.\nThe AI is implemented using Minimax Algorithm and Alpha Beta Pruning with varying heuristics.\n\n## The Game\n\nThe game is being developed using PyGame. The game is played by two players. The player who gets four consecutive pieces in a row, column or diagonal wins the game. The game is played on a 6x7 board. The game is played by two players. The player who gets four consecutive pieces in a row, column or diagonal wins the game. The game is played on a 6x7 board.\n\n\n## The AI - Minimax Algorithm with Alpha Beta Pruning\nThe AI is implemented using Minimax Algorithm with Alpha Beta Pruning Algorithm. Minimax is a decision rule used in artificial intelligence, decision theory, game theory, statistics, and philosophy for minimizing the possible loss for a worst case (maximum loss) scenario. When dealing with gains, it is referred to as "maximin"\u2014to maximize the minimum gain. Originally formulated for two-player zero-sum game theory, covering both the cases where players take alternate moves and those where they make simultaneous moves, it has also been extended to more complex games and to general decision-making in the presence of uncertainty.\n\n\n## Implementation\nThe following code snippet shows the implementation of the Minimax Algorithm with Alpha Beta Pruning Algorithm.\n',r='import pygame\nimport sys\n\n# Constants\nROW_COUNT = 6\nCOLUMN_COUNT = 7\nSQUARE_SIZE = 100\nWINDOW_WIDTH = COLUMN_COUNT * SQUARE_SIZE\nWINDOW_HEIGHT = (ROW_COUNT + 1) * SQUARE_SIZE  # +1 for the header row\n\n# Colors\nBLACK = (0, 0, 0)\nWHITE = (255, 255, 255)\nBLUE = (0, 0, 255)\nRED = (255, 0, 0)\nYELLOW = (255, 255, 0)\n\n# Initialize Pygame\npygame.init()\n\n# Create the game window\nwindow = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))\n\n# Initialize the game board\nboard = [[0] * COLUMN_COUNT for _ in range(ROW_COUNT)]\n\n# Initialize player and game state variables\ncurrent_player = 1\ngame_state = "playing"  # "playing", "won", or "restart"\n\n# Initialize posx with a default value\nposx = WINDOW_WIDTH // 2\n\n# Function to draw the board\ndef draw_board(board):\n    for row in range(ROW_COUNT):\n        for col in range(COLUMN_COUNT):\n            pygame.draw.rect(window, BLUE, (col * SQUARE_SIZE, (row + 1) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))\n            pygame.draw.circle(window, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 2, (row + 1) * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)\n            if board[row][col] == 1:\n                pygame.draw.circle(window, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, (row + 1) * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)\n            elif board[row][col] == 2:\n                pygame.draw.circle(window, YELLOW, (col * SQUARE_SIZE + SQUARE_SIZE // 2, (row + 1) * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)\n\n# Function to drop a piece in a column\ndef drop_piece(board, row, col, player):\n    board[row][col] = player\n\n# Function to check if a move is valid\ndef is_valid_move(board, col):\n    return board[0][col] == 0\n\n# Function to get the next available row in a column\ndef get_next_open_row(board, col):\n    for r in range(ROW_COUNT - 1, -1, -1):\n        if board[r][col] == 0:\n            return r\n\n# Function to check if a player has won\ndef check_win(board, player):\n    # Check horizontally\n    for row in range(ROW_COUNT):\n        for col in range(COLUMN_COUNT - 3):\n            if board[row][col] == player and board[row][col + 1] == player and board[row][col + 2] == player and board[row][col + 3] == player:\n                return True\n\n    # Check vertically\n    for row in range(ROW_COUNT - 3):\n        for col in range(COLUMN_COUNT):\n            if board[row][col] == player and board[row + 1][col] == player and board[row + 2][col] == player and board[row + 3][col] == player:\n                return True\n\n    # Check diagonally (from bottom-left to top-right)\n    for row in range(3, ROW_COUNT):\n        for col in range(COLUMN_COUNT - 3):\n            if board[row][col] == player and board[row - 1][col + 1] == player and board[row - 2][col + 2] == player and board[row - 3][col + 3] == player:\n                return True\n\n    # Check diagonally (from bottom-right to top-left)\n    for row in range(3, ROW_COUNT):\n        for col in range(3, COLUMN_COUNT):\n            if board[row][col] == player and board[row - 1][col - 1] == player and board[row - 2][col - 2] == player and board[row - 3][col - 3] == player:\n                return True\n\n# Function to display a message on the screen\ndef draw_message(message):\n    font = pygame.font.Font(None, 36)\n    text = font.render(message, True, WHITE)\n    window.blit(text, (20, 10))\n\n# Draw the initial board\ndraw_board(board)\npygame.display.update()\n\n# Main game loop\nwhile True:\n    for event in pygame.event.get():\n        if event.type == pygame.QUIT:\n            pygame.quit()\n            sys.exit()\n\n        if game_state == "playing":\n            if event.type == pygame.MOUSEMOTION:\n                pygame.draw.rect(window, BLACK, (0, 0, WINDOW_WIDTH, SQUARE_SIZE))\n                posx = event.pos[0]\n                posy = SQUARE_SIZE // 2\n                pygame.draw.circle(window, RED if current_player == 1 else YELLOW, (posx, posy), SQUARE_SIZE // 2 - 5)\n                pygame.display.update()\n\n            if event.type == pygame.MOUSEBUTTONDOWN:\n                pygame.draw.rect(window, BLACK, (0, 0, WINDOW_WIDTH, SQUARE_SIZE))\n                col = posx // SQUARE_SIZE\n\n                if is_valid_move(board, col):\n                    row = get_next_open_row(board, col)\n                    drop_piece(board, row, col, current_player)\n\n                    if check_win(board, current_player):\n                        winner_message = f"Player {current_player} wins! Press Space to restart."\n                        draw_message(winner_message)\n                        game_state = "won"\n                    else:\n                        current_player = 3 - current_player\n                    draw_board(board)\n                    pygame.display.update()\n\n            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:\n                game_state = "restart"\n\n        elif game_state == "won":\n            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:\n                board = [[0] * COLUMN_COUNT for _ in range(ROW_COUNT)]\n                current_player = 1\n                pygame.draw.rect(window, BLACK, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))\n                draw_board(board)\n                pygame.display.update()\n                game_state = "playing"\n\n        elif game_state == "restart":\n            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:\n                board = [[0] * COLUMN_COUNT for _ in range(ROW_COUNT)]\n                current_player = 1\n                pygame.draw.rect(window, BLACK, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))\n                draw_board(board)\n                pygame.display.update()\n                game_state = "playing"\n\n'}}]);
//# sourceMappingURL=936.c4d71c78.chunk.js.map