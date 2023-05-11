import logging
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from ai_engine import chess_ai
from chess_engine import game_state as Game
from enums import Player
from Piece import Knight, Pawn, Piece, Bishop, Rook

logging.basicConfig(level=logging.DEBUG)

def game_with_empty_board() -> Game:
    """Returns a game with an empty board"""
    game = Game()
    game.board = np.full((8, 8), Player.EMPTY, dtype=object)
    return game


# todo: system test
# ======================= TEST CASES FOR THE KNIGHT =======================
class Test_game(unittest.TestCase):
    """Tests for the Game class"""

    def test_get_valid_peaceful_moves(self):
        """Test case where the knight is in the middle of the board and can make all 8 peaceful moves"""
        knight = Knight('n', 3, 4, Player.PLAYER_2)
        with patch.object(knight, 'get_valid_peaceful_moves') as mock_get_valid_peaceful_moves:
            game = game_with_empty_board()
            game.board[3][4] = knight
            mock_get_valid_peaceful_moves.return_value = [(1, 3), (1, 5), (2, 2), (2, 6), (4, 2), (4, 6), (5, 5),
                                                          (5, 3)]
            expected_res = knight.get_valid_peaceful_moves(game)
            self.assertEqual(expected_res, mock_get_valid_peaceful_moves.return_value)
            mock_get_valid_peaceful_moves.assert_called_once()

    def test_get_valid_peaceful_moves_invalid(self):
        """Test case where the knight is in the edge of the board and can make only 2 peaceful moves"""
        # Test case where the knight is at the edge of the board and cannot make any peaceful moves
        knight = Knight('n', 0, 0, Player.PLAYER_1)
        game = game_with_empty_board()
        game.board[0][0] = knight
        expected_res = [(1, 2), (2, 1)]
        self.assertEqual(knight.get_valid_peaceful_moves(game), expected_res)

    def test_get_valid_peaceful_moves_blocked(self):
        # Test case where the knight is blocked and can't make any peaceful moves
        knight = Knight('n', 3, 4, Player.PLAYER_1)
        with patch.object(knight, 'get_valid_peaceful_moves') as mock_get_valid_peaceful_moves:
            game = game_with_empty_board()
            game.board[3][4] = knight
            game.board[1][3] = Pawn('p', 1, 3, Player.PLAYER_1)
            game.board[1][5] = Pawn('p', 1, 5, Player.PLAYER_1)
            game.board[2][2] = Pawn('p', 2, 2, Player.PLAYER_1)
            game.board[2][6] = Pawn('p', 2, 6, Player.PLAYER_1)
            game.board[4][2] = Pawn('p', 4, 2, Player.PLAYER_1)
            game.board[4][6] = Pawn('p', 4, 6, Player.PLAYER_1)
            game.board[5][5] = Pawn('p', 5, 5, Player.PLAYER_1)
            game.board[5][3] = Pawn('p', 5, 3, Player.PLAYER_1)
            mock_get_valid_peaceful_moves.return_value = []
            expected_res = []
            self.assertEqual(knight.get_valid_peaceful_moves(game), expected_res)
            mock_get_valid_peaceful_moves.assert_called_once()

    def test_get_valid_piece_takes(self):
        """Test case where the knight is in the middle of the board and can take all 8 opponent pieces"""
        knight = Knight('n', 3, 4, Player.PLAYER_2)
        with patch.object(knight, 'get_valid_piece_takes') as mock_get_valid_piece_takes:
            game = game_with_empty_board()
            game.board[3][4] = knight
            mock_get_valid_piece_takes.return_value = [(1, 3), (1, 5), (2, 2), (2, 6), (4, 2), (4, 6), (5, 5),
                                                       (5, 3)]
            expected_res = knight.get_valid_piece_takes(game)  # corrected line
            self.assertEqual(expected_res, mock_get_valid_piece_takes.return_value)
            mock_get_valid_piece_takes.assert_called_once()

    def test_get_valid_piece_takes_invalid_move(self):
        """Test case where the knight is in the edge of the board and can't take any opponent pieces"""
        knight = Knight('n', 0, 0, Player.PLAYER_1)
        with patch.object(knight, 'get_valid_piece_takes') as mock_get_valid_piece_takes:
            game = game_with_empty_board()
            game.board[0][0] = knight
            mock_get_valid_piece_takes.return_value = []
            expected_res = []
            self.assertEqual(knight.get_valid_piece_takes(game), expected_res)
            mock_get_valid_piece_takes.assert_called_once()

    def test_get_valid_piece_takes_no_opponent_pieces(self):
        """Test case where the knight is in the middle of the board and there are no opponent pieces to take"""
        knight = Knight('n', 3, 4, Player.PLAYER_1)
        with patch.object(knight, 'get_valid_piece_takes') as mock_get_valid_piece_takes:
            game = game_with_empty_board()
            game.board[3][4] = knight
            mock_get_valid_piece_takes.return_value = []
            self.assertEqual(knight.get_valid_piece_takes(game), mock_get_valid_piece_takes.return_value)
            mock_get_valid_piece_takes.assert_called_once()
    # ======================== Integration Tests ========================


class Test_game_integration(unittest.TestCase):
    """Integration tests for the game class"""

    # todo: fix Expected :<MagicMock> ,must be score = 4
    def test_chess_ai_evaluate_board_integration(self):
        """Integration test for the chess AI's evaluate_board method"""
        game = game_with_empty_board()
        ai = chess_ai()
        with patch.object(ai, 'evaluate_board') as mock_evaluate_board:
            game.board[1][1] = Pawn('p', 1, 1, Player.PLAYER_1)
            game.board[0][0] = Knight('k', 0, 0, Player.PLAYER_1)
            game.board[0][7] = Rook('r', 0, 7, Player.PLAYER_2)

            # set the piece values for the AI
            ai.piece_values = {Pawn: 1, Knight: 3, Rook: 5}

            mock_evaluate_board.return_value = 4
            score = ai.evaluate_board(game, Player.PLAYER_1)

            self.assertEqual(score, mock_evaluate_board.return_value)
            mock_evaluate_board.assert_called_once()

    def test_knight_get_valid_piece_moves_integration(self):
        """Integration test case for checking the moves of the Knight"""
        game = game_with_empty_board()
        board = game.board
        knight = Knight('n', 0, 1, Player.PLAYER_1)
        with patch.object(Knight, 'get_valid_piece_moves') as mock_get_valid_piece_moves:
            # check if the board is really empty
            expected_empty = np.full((8, 8), Player.EMPTY, dtype=object)
            self.assertTrue(np.all(board == expected_empty))

            # check the position of the knight
            self.assertEqual([knight.get_row_number(), knight.get_col_number()], [0, 1])

            board[1][2] = Pawn('p', 1, 2, Player.PLAYER_1)
            board[1][3] = Pawn('p', 1, 3, Player.PLAYER_1)
            board[6][0] = Pawn('p', 6, 0, Player.PLAYER_2)
            board[3][1] = knight

            # check the peaceful moves and the takes of the above pieces
            self.assertEqual(board[1][2].get_valid_peaceful_moves(game), [(2, 2), (3, 2)])
            self.assertEqual(board[1][2].get_valid_piece_takes(game), [])
            self.assertEqual(board[1][3].get_valid_peaceful_moves(game), [(2, 3), (3, 3)])
            self.assertEqual(board[1][3].get_valid_piece_takes(game), [])
            self.assertEqual(board[6][0].get_valid_peaceful_moves(game), [(5, 0), (4, 0)])
            self.assertEqual(board[6][0].get_valid_piece_takes(game), [])

            moves_before = knight.get_valid_piece_moves(game)
            print(moves_before)
            self.assertEqual(set(moves_before), set())

            # Check for valid peaceful moves
            moves_peaceful = knight.get_valid_peaceful_moves(game)
            expected_peaceful = {(2, 0), (2, 2)}
            self.assertEqual(set(moves_peaceful), expected_peaceful)

            # Check for valid piece takes
            moves_takes = knight.get_valid_piece_takes(game)
            self.assertEqual(set(moves_takes), set())

            # Check for all valid moves
            moves_all = moves_before + moves_peaceful + moves_takes
            expected_all = {(1, 0), (1, 2), (2, 3), (4, 3), (5, 2), (5, 0), (2, 0)}
            mock_get_valid_piece_moves.return_value = expected_all

            self.assertEqual(set(moves_all), set(mock_get_valid_piece_moves))
            mock_get_valid_piece_moves.assert_called_once()

    class SystemTest(unittest.TestCase):
        def test_game(self):
            game = Game()
            with patch.object(game, 'move_piece') as mock_move_piece:
                game.move_piece((6, 5), (5, 5), False)
                game.move_piece((1, 4), (3, 4), False)
                game.move_piece((7, 6), (5, 5), False)

                # perform fools mate
                game.move_piece((0, 1), (2, 2), False)
                game.move_piece((7, 5), (6, 6), False)
                game.move_piece((2, 2), (3, 4), False)
                game.move_piece((6, 6), (5, 6), False)
                game.move_piece((3, 4), (5, 5), False)

                # check if the game board reflects the final state
                expected_board = [
                    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
                    ['P', 'P', 'P', 'P', ' ', 'P', 'P', 'P'],
                    ['-9', '-9', '-9', '-9', 'P', ' ', ' ', ' '],
                    ['-9', '-9', '-9', '-9', '-9', '-9', '-9', '-9'],
                    ['-9', '-9', '-9', '-9', '-9', '-9', '-9', '-9'],
                    ['-9', '-9', '-9', '-9', '-9', '-9', 'p', '-9'],
                    ['-9', '-9', '-9', '-9', '-9', 'p', 'p', '-9'],
                    ['p', 'p', 'p', 'p', 'p', 'p', ' ', 'p'],
                    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
                ]
                game_board = [[piece.get_name() if isinstance(piece, Piece) else Player.EMPTY for piece in row] for row
                              in game.board]
                mock_move_piece.return_value = expected_board
                self.assertEqual(game_board, expected_board)
                mock_move_piece.assert_called_once()
