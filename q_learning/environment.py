#1.首先定义好动作的宏
#2.定义初始化函数，确定起点和终点
#3.定义重置函数，确定每次测试时都回到起点
#4.定义行动函数
#5.定义显示函数，能看到行走轨迹

class GridWorld:

    UP = 0

    DOWN = 1

    LEFT = 2

    RIGHT = 3

    def __init__(self):

        self.rows = 4
        self.cols = 4

        self.strat_state = (0,0)
        self.goal_state = (3,3)

        self.state = self.strat_state

    def reset(self):

        self.state = self.strat_state

        return self.state

    def step(self,action):

        row,col = self.state

        if action == self.UP:
            row -= 1
        elif action == self.DOWN:
            row += 1
        elif action == self.LEFT:
            col -= 1
        elif action == self.RIGHT:
            col += 1
        else:
            raise ValueError("Invalid action")

        row = max(0,min(row,self.rows - 1))
        col = max(0,min(col,self.cols - 1))

        next_state = (row,col)

        if next_state == self.goal_state:

            reward = 0
            done = True

        else:
            reward = -1
            done = False

        self.state = next_state

        return next_state,reward,done

    def render(self):

        for row in range(self.rows):

            line = []

            for col in range(self.cols):

                position = (row, col)

                if position == self.state:
                    line.append("A")

                elif position == self.goal_state:
                    line.append("G")

                else:
                    line.append(".")

            print(" ".join(line))

        print()

    
