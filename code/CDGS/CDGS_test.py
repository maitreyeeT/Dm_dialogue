from collections import defaultdict

class CDGS():

    def __init__(self, blackboard, start_symbol = "S"):
        super(CDGS, self).__init__()
        self.blackboard = blackboard
        self.start_symbol = start_symbol
        self.end_symbol = None

    def cooperate(self):
        agent1 = self.agentx()
        agent2 = self.agenty()
        protocol = 2
        end_state = lambda x: x == 'null'
        while not end_state:
            n = 1
            x1,x2,x3 = [x1x2x3 for x1x2x3 in agent1]
            print(x1,x2,x3)
            self.blackboard.append(self.start_symbol)
            self.blackboard.append([x1[key] for key in x1 if key == self.start_symbol])
            self.blackboard.append([x2[key] for key in x2 if key == 'G'])
            self.blackboard.append([x3[key] for key in x3 if key == 'Q'])

            for y1y2y3y4 in agent2:
                y1, y2, y3, y4 = y1y2y3y4

        return(self.blackboard)

    def expectation(self):
        pass

    #parse the rules here for agent 1 and agent 2
    def agentx(self):
        x1 = {'S': ['G','Q','Agenty']}
        x2 = {'G': ['hello']}
        x3 = {'Q': ['how are you doing today?']}
        yield (x1,x2,x3)

    def agenty(self):
        y1 = {'Agenty': 'GA'}
        y2 = {'G': 'hello'}
        y3 = {'Q': 'I am doing good'}
        y4 = {'Delim' : 'null'}
        yield (y1,y2,y3,y4)

if __name__ == '__main__':
    pepperboard = []
    cdgs = CDGS(pepperboard)
    cooperate = cdgs.cooperate()
    print(cooperate)

