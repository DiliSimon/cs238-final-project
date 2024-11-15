from InvasiveEnvironment import InvasiveEnvironment
import os
import numpy as np
from rlglue.types import Action




if __name__ == "__main__":
    Exp = InvasiveEnvironment(simulationParameterObj=None, actionParameterObj=None, Bad_Action_Penalty=-10000,fixedStartState=False, nbrReaches=7,
            habitatSize=4, seed=1)
    Exp.env_init()
    Exp.env_start()
    print("initial state", Exp.state)
    action = Action(numInts=7)
    action.intArray = [1,1,1,1,1,1,4]
    print(action.intArray)
    for i in range(100):
        Exp.env_step(action)
        print(Exp.state)

    print(sum(Exp.state[Exp.state == 2])/2)
    print(sum(Exp.state[Exp.state == 1]))
