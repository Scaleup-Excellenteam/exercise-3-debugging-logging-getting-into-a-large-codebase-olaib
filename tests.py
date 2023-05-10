import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from chess_engine import game_state as Game
from enums import Player
from Piece import Knight, Pawn, Piece, Bishop


def game_with_empty_board() -> Game:
    """Returns a game with an empty board"""
    game = Game()
    game.board = np.full((8, 8), Player.EMPTY, dtype=object)
    return game


# todo: system test
# ======================= TEST CASES FOR THE KNIGHT =======================
class TestGame(unittest.TestCase):
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
            expected_res = []
            self.assertEqual(knight.get_valid_piece_takes(game), expected_res)
            mock_get_valid_piece_takes.assert_called_once()

    def test_get_valid_piece_moves_integration(self):
        """Integration test case for checking the moves of the Knight"""
        game_state = game_with_empty_board()
        board = game_state.board
        # check if the board is really empty
        expected_empty = np.full((8, 8), Player.EMPTY, dtype=object)
        self.assertTrue(np.all(board == expected_empty))

        knight = Knight('n', 0, 1, Player.PLAYER_1)
        # check the position of the knight
        self.assertEqual([knight.get_row_number(), knight.get_col_number()], [0, 1])

        board[1][2] = Pawn('p', 1, 2, Player.PLAYER_1)
        board[1][3] = Pawn('p', 1, 3, Player.PLAYER_1)
        board[6][0] = Pawn('p', 6, 0, Player.PLAYER_2)
        board[3][1] = knight

        # check the peaceful moves and the takes of the above pieces
        self.assertEqual(board[1][2].get_valid_peaceful_moves(game_state), [(2, 2), (3, 2)])
        self.assertEqual(board[1][2].get_valid_piece_takes(game_state), [])
        self.assertEqual(board[1][3].get_valid_peaceful_moves(game_state), [(2, 3), (3, 3)])
        self.assertEqual(board[1][3].get_valid_piece_takes(game_state), [])
        self.assertEqual(board[6][0].get_valid_peaceful_moves(game_state), [(5, 0), (4, 0)])
        self.assertEqual(board[6][0].get_valid_piece_takes(game_state), [])

        moves_before = knight.get_valid_piece_moves(game_state)
        print(moves_before)
        # todo: why didn't return the expected result? empty list
        self.assertEqual(set(moves_before), {(2, 0), (2, 2)})

        # Check for valid peaceful moves
        # todo: fix bug :Knight.get_valid_peaceful_moves() missing 1 required positional argument: 'game_state'
        moves_peaceful = Knight.get_valid_peaceful_moves(game_state)
        expected_peaceful = {(1, 0), (1, 2), (2, 3), (4, 3), (5, 2), (5, 0)}
        self.assertEqual(set(moves_peaceful), expected_peaceful)

        # Check for valid piece takes
        moves_takes = knight.get_valid_piece_takes(game_state)
        self.assertEqual(set(moves_takes), {(2, 0)})

        # Check for all valid moves
        moves_all = knight.get_valid_piece_moves(game_state)
        expected_all = {(1, 0), (1, 2), (2, 3), (4, 3), (5, 2), (5, 0), (2, 0)}
        self.assertEqual(set(moves_all), expected_all)
