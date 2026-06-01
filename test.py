from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.render()

# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    from random import choice
    # env.render()
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

env.reset()
# Play as the first agent against default "random" agent.
env.run([my_agent, "random"])
env.render(mode="human", width=500, height=450)

# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()

env.play([None, "negamax"], width=500, height=450)