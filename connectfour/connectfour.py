import sys
import random
import copy

FIELD_EMPTY = 0
FIELD_A = 1
FIELD_B = 2
SIZE_X = 7
SIZE_Y = 6
WORST_SCORE = -10000
BEST_SCORE = 10000

def enemy_of(player):
    return FIELD_B if player == FIELD_A else FIELD_A

class Game(object):
    field = []

    def __init__(self):
        self.field = []
        for i in range(SIZE_X * SIZE_Y):
            self.field.append(FIELD_EMPTY)


    def get_field(self, x, y):
        return self.field[y * SIZE_X + x]


    def set_field(self, x, y, v):
        self.field[y * SIZE_X + x] = v

    def print_state(self):
        for x in range(SIZE_X):
            sys.stdout.write(str(x) + " ")
        print("")
        print("- - - - - - -")
        for y in range(SIZE_Y-1, -1, -1):
            for x in range(SIZE_X):
                c = '.'
                if self.get_field(x,y) != FIELD_EMPTY:
                    c = str(self.get_field(x,y))
                sys.stdout.write(c + " ")
            print("")

        print("- - - - - - -")
        for x in range(SIZE_X):
            sys.stdout.write(str(x) + " ")
        print("")


    def can_drop(self, column):
        return self.get_field(column,SIZE_Y-1) == FIELD_EMPTY


    def drop_piece(self, column, player):
        row = 0
        while row < SIZE_Y and self.get_field(column,row) != FIELD_EMPTY:
            row = row + 1
        if row >= SIZE_Y:
            raise Exception("Full Column")
        else:
            self.set_field(column, row, player)


    def find_line_len(self, sx, sy, dx, dy, who):
        x = sx
        y = sy
        for i in xrange(4):
            if x < 0 or x >= SIZE_X or y < 0 or y >= SIZE_Y or self.field[y * SIZE_X + x] != who:
                return i
            x = x + dx
            y = y + dy
        return 4


    def get_line_lens(self, who):
        all_lens = []
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if self.field[y * SIZE_X + x] == who:
                    all_lens.extend([
                        self.find_line_len(x, y, 0, 1, who),
                        self.find_line_len(x, y, 1, 0, who),
                        self.find_line_len(x, y, 1, 1, who),
                        self.find_line_len(x, y,-1, 1, who),
                    ])
        return all_lens


    def find_longest_line(self, who):
        longest = 0
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if self.field[y * SIZE_X + x] == who:
                    new_longest = max([
                        self.find_line_len(x, y, 0, 1, who),
                        self.find_line_len(x, y, 1, 0, who),
                        self.find_line_len(x, y, 1, 1, who),
                        self.find_line_len(x, y,-1, 1, who),
                    ])

                    if new_longest > longest:
                        longest = new_longest
                        if longest >= 4:
                            return longest
        return longest


    def find_winner(self):
        for winner in range(1,3):
            longest_line = self.find_longest_line(winner)
            if longest_line >= 4:
                return winner
        return FIELD_EMPTY


    def is_draw(self):
        for x in range(SIZE_X):
            if self.get_field(x, SIZE_Y - 1) == FIELD_EMPTY:
                return False
        return True

    def score(self, scored_player):
        enemy = enemy_of(scored_player)
        my_lens = self.get_line_lens(scored_player)

        if len(my_lens) > 0 and max(my_lens) >= 4:
            return BEST_SCORE

        enemy_lens = self.get_line_lens(enemy)
        if len(enemy_lens) > 0 and max(enemy_lens) >= 4:
            return WORST_SCORE

        my_squared = map(lambda x: x**2, my_lens)
        enemy_squared = map(lambda x: x**2, enemy_lens)
        return sum(my_squared) - sum(enemy_squared)

    def undo(self, column):
        for y in xrange(SIZE_Y):
            if self.get_field(column, SIZE_Y - 1 - y) != FIELD_EMPTY:
                self.set_field(column, SIZE_Y - 1 - y, FIELD_EMPTY)
                break

    def random_move(self, who):
        column = 0
        for i in range(20):
            column = random.randint(0, SIZE_X - 1)
            if self.can_drop(column):
                self.drop_piece(column, who)
                return

        for i in range(SIZE_X):
            if self.can_drop(column):
                self.drop_piece(column, who)
                return

    def clever_move(self, who, lookahead=1):
        enemy = enemy_of(who)
        best_column = 0
        best_score = -10000000
        shuffled_columns = []
        shuffled_columns.extend(range(SIZE_X))
        random.shuffle(shuffled_columns)
        for column in shuffled_columns:
            if not self.can_drop(column):
                continue
            self.drop_piece(column, who)
            other_move = -1
            if lookahead > 0:
                other_move = self.clever_move(enemy, lookahead-1)
            score = self.score(who)

            if other_move >= 0:
                self.undo(other_move)

            self.undo(column)

            if score > best_score:
                best_column = column
                best_score = score

        if self.can_drop(column):
            self.drop_piece(best_column, who)
            return best_column
        else:
            return -1
