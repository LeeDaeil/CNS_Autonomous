from CNS.Auto_cont.UDP_network import UDP_net

class Aurorun:
    def __init__(self): # 0: run, 1: initial, 2: stop
        self.UDP = UDP_net(mode=False)

    def run(self):
        self.UDP.make_sock()
        self.UDP.send_message('0')
        return print(str(self.UDP.read_data()))

    def initial(self):
        self.UDP.make_sock()
        self.UDP.send_message('1')
        return print(str(self.UDP.read_data()))

    def stop(self):
        self.UDP.make_sock()
        self.UDP.send_message('2')
        return print(str(self.UDP.read_data()))

    def close(self):
        self.UDP.close_sock()
        return print('Close socket....')


# if __name__ == "__main__":
    '''
    UDP = UDP_net(mode= False, timemode=False)

    while True:
        #
        #CNS part
        #
        UDP.make_sock()
        data = UDP.read_data()
        Mouse = Mou_cont()
        if int(data) == 0:
            Mouse.auto_run()
            UDP.send_message('Run')
        if int(data) == 1:
            Mouse.auto_move()
            UDP.send_message('Initial condition Done')
        elif int(data) == 2:
            UDP.send_message('Stop program')
            break
        else:
            UDP.send_message('Worrong signal')

    UDP.close_sock()
    '''

    '''
    while True:
        #
        #Remote com
        #
        UDP.make_sock()
        Ordder = input() # 0: Run 1: Initial, 2: Stop

        UDP.send_message(Ordder)
        if Ordder == '2':
            print('Stop program')
            break
        Date = UDP.read_data()

        print(str(Date))

    UDP.close_sock()
    '''