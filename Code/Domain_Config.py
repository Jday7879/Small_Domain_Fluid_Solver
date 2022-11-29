Reactor = dict(
    x_length = 78e-3,
    y_length = 20e-3, # Influent is placed on the y axis
    z_length = 9e-3,
    influent_width = 3e-3 # Note influent width must be smaller than y
)

Retention = dict(
    time = 1,
    units = 'm' # Choose units for retention time (d,h,m): d - days; h - hours; m - mins.
)


if __name__ == '__main__':
    if Reactor['influent_width'] < Reactor['y_length']:
        print('Checking Input Variables are correct')
        print('\n X Length = {}m \n Y Length = {}m \n Z Length = {}m \n Influent Width = {}m'.format(Reactor['x_length'],Reactor['y_length'],Reactor['z_length'],Reactor['influent_width']))
        if Retention['units'] == 'd':
            units = 'Day(s)'
        elif Retention['units'] =='h':
            units = 'Hour(s)'
        elif Retention['units'] == 'm':
            units = 'Min(s)'

        print('\n Retention Time {}'.format(Retention['time'])+' '+ units)
    else:
        print('Influent width must be smaller than y length')