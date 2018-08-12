from pynput.mouse import Button, Controller
from time import sleep

class Mou_cont:
    def __init__(self):
        self.mouse = Controller()
        self.delay = 0.1

    def click_(self):
        self.mouse.click(Button.left, 1)
        sleep(self.delay)

    def position(self,x,y):
        self.mouse.position = (x,y)
        sleep(self.delay)

    def drag(self, x1, y1, x2, y2):
        self.position(x1, y1)
        self.mouse.press(Button.left)
        sleep(self.delay)

        self.position(x2, y2)
        self.mouse.release(Button.left)
        sleep(self.delay)

    def auto_move(self):
        self.position(184, 282)
        self.click_()

        self.position(480, 279)
        self.click_()

        self.drag(959, 633, 959, 744)

        self.position(683, 795)
        self.click_()

        self.position(909, 602)
        self.click_()

        return print('Done') # done

    def auto_run(self):
        self.position(188,250)
        self.click_()

if __name__ == "__main__":
    Mouse = Mou_cont()
    Mouse.auto_move()