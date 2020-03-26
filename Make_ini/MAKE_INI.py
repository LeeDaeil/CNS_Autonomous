def make_DCS_com_pid():
    with open('want_val.txt', 'r') as f:
        val_list = f.read().split('\n')

    print(len(val_list), val_list)

    with open('db.txt', 'r') as f:
        db_dict = {}
        while True:
            temp_ = f.readline().split('\t')
            if temp_[0] == '':
                break
            else:
                db_dict[temp_[0]] = temp_[1]

    with open('DCSCommPid.ini', 'w') as f_pid:
        nub_line = 0
        for val in val_list:
            if '#' in val:
                pass
            else:
                if nub_line == 0:
                    f_pid.write('{}\t{}\t{}'.format(nub_line, val, db_dict[val]))
                else:
                    f_pid.write('\n{}\t{}\t{}'.format(nub_line, val, db_dict[val]))
                nub_line += 1
    return 0

make_DCS_com_pid()