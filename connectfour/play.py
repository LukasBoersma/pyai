import connectfour

import samples

import random
from multiprocessing.pool import Pool

def play(ai, count=300, clever=True):
    wins = [0,0]
    invalid = 0

    win_percent = 0

    show_gamestate = False

    for i in range (count):
        game = connectfour.Game()
        while game.find_winner() == connectfour.FIELD_EMPTY and not game.is_draw():
            move = ai.move(samples.field_vector(game))
            try:
                game.drop_piece(move,1)
            except Exception:
                invalid += 1
                break

            if show_gamestate:
                print("Neural AI places a stone")
                game.print_state()


            if game.find_winner() != connectfour.FIELD_EMPTY:
                break

            if clever:
                game.clever_move(2, 1)
            else:
                game.random_move(2)

            if show_gamestate:
                print("Traditional AI places a stone")
                print("")
                game.print_state()
                print("============================")

        if not game.is_draw():
            #print("Draw! Nobody wins.")
        #else:
            #print("And the winner is " + str(game.find_winner()))
            if game.find_winner() == 1:
                wins[0] += 1
            elif game.find_winner() == 2:
                wins[1] += 1
    win_percent = 100*float(wins[0]) / (i+1)
    #print("Current score: (Player 1) " +str(wins[0]) + " : " + str(wins[1])    + " (Player 2). Invalid: " + str(invalid) + "    [" + str(win_percent) + "%]")
    print("Score: (Player 1) %d : %d (Player 2). Invalid: %d, Wins: %.2f" % (wins[0], wins[1], invalid, win_percent))
    return win_percent

pool_size = 8
batch_count = pool_size# * 3
pool = Pool(pool_size)

def play_parallel(ai, count=300, clever=True):

    results = []
    for i in range(batch_count):
        results.append(pool.apply_async(play, (count/batch_count,clever)))

    outputs = []
    for r in results:
        outputs.append(r.get(1000))

    avg_score = sum(outputs) / len(outputs)
    print("Avg score: %.2f"  % avg_score)
    return avg_score

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
