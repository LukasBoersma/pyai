import connectfour
import copy
import random
import gzip
from multiprocessing.pool import Pool


def score_vector(game, player):
    enemy = connectfour.enemy_of(player)
    scores = []
    for column in range(connectfour.SIZE_X):
        if not game.can_drop(column):
            scores.append(connectfour.WORST_SCORE)
        else:
            future = copy.deepcopy(game)
            future.drop_piece(column, player)
            future.clever_move(enemy, 0)
            scores.append(future.score(player))

    min_score = min(scores)
    scores[:] = map(lambda x: x-min_score, scores)
    score_sum = sum(scores)+0.0001
    scores[:] = map(lambda x: x/score_sum, scores)
    #max_score = max(scores)
    #scores[:] = map(lambda x: 1 if x >= max_score else 0, scores)
    return scores


def field_vector(game):
    f = map(lambda s: [
        1 if s == connectfour.FIELD_EMPTY else 0,
        1 if s == connectfour.FIELD_A else 0,
        1 if s == connectfour.FIELD_B else 0
    ], game.field)
    return [t for s in f for t in s]


def create_training_data(count):
    samples = []
    game_count = 0
    while len(samples) < count:
        game = connectfour.Game()
        game_count += 1
        #print("game %d, count %d" % (game_count, len(samples)))

        while game.find_winner() == connectfour.FIELD_EMPTY and not game.is_draw():
            samples.append((field_vector(game), score_vector(game, 1)))
            if len(samples) >= count:
                break
            # Player 1 move
            game.clever_move(1)
            if game.find_winner() != connectfour.FIELD_EMPTY or game.is_draw():
                break
            # Player 2 move
            game.clever_move(2)
    random.shuffle(samples)
    print("generated %d samples" % count)
    return samples

def create_training_parallel(count):
    pool_size = 8
    batch_count = pool_size * 5
    pool = Pool(pool_size)
    print("generating")
    results = []
    for i in range(batch_count):
        results.append(pool.apply_async(create_training_data, (count/batch_count,)))

    pool.close()
    pool.join()
    print("concatenating")

    output = []
    for r in results:
        output.extend(r.get(1000))
    return output

def save(samples, filename):
    f = gzip.open(filename, 'w')
    f.write(str(samples))
    f.close()

def load(filename):
    f = gzip.open(filename, 'r')
    samples = eval(f.read())
    f.close()
    return samples