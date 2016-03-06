import connectfour

def clever_play():
    game = connectfour.Game()
    while game.find_winner() == connectfour.FIELD_EMPTY and not game.is_draw():
        game.clever_move(1, 1)
        game.print_state()
        if game.find_winner() != connectfour.FIELD_EMPTY or game.is_draw():
            break
        game.clever_move(2, 1)
        game.print_state()

    print("============================")
    if game.is_draw():
        print("Draw! Nobody wins.")
        return connectfour.FIELD_EMPTY
    else:
        winner = game.find_winner()
        print("And the winner is " + str(winner))
        return winner

def random_play():
    game = connectfour.Game()
    while game.find_winner() == connectfour.FIELD_EMPTY and not game.is_draw():
        game.random_move(1)
        game.print_state()
        if game.find_winner() != connectfour.FIELD_EMPTY or game.is_draw():
            break
        game.random_move(2)
        game.print_state()

    print("============================")
    if game.is_draw():
        print("Draw! Nobody wins.")
        return connectfour.FIELD_EMPTY
    else:
        winner = game.find_winner()
        print("And the winner is " + str(winner))
        return winner

def human_play():
    game = connectfour.Game()
    while game.find_winner() == connectfour.FIELD_EMPTY and not game.is_draw():
        game.clever_move(1)
        game.print_state()
        if game.find_winner() != connectfour.FIELD_EMPTY or game.is_draw():
            break
        column = int(input("Which column? "))
        game.drop_piece(column, 2)
        game.print_state()

    print("============================")
    if game.is_draw():
        print("Draw! Nobody wins.")
        return connectfour.FIELD_EMPTY
    else:
        winner = game.find_winner()
        print("And the winner is " + str(winner))
        return winner
