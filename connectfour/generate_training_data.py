
import samples

sizes = [500, 1500, 3000,5000]

for s in sizes:
    print("generate" + str(s))
    td = samples.create_training_data_one_per_game(s)

    samples.save(td,"data" +str(s) + ".txt")
