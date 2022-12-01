# -*- coding: utf-8 -*-

# Copyright 2021 Jordan Day
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
import solver
from Plotting_thesis import *
import Domain_Config

print(Domain_Config.Retention['time'])

def boundary_init(influent_width):
    global flux, baffle_length, baffle_pairs,Ly, Lx,dx, dy
    psi = np.zeros((nx + 1, ny + 1))
    boundary = np.zeros((nx + 1, ny + 1))  # We set this to 1 on all boundary points
    boundary[0, :] = 1
    boundary[-1, :] = 1
    boundary[:, 0] = 1
    boundary[:, -1] = 1

    psi[0, 0:ny + 3] = flux
    psi[:, -1] = flux

    channel_width = Ly / (baffle_pairs * 2 + 1)
    if channel_width > influent_width:
        in_out_points = influent_width / dy
    else:  # Default influent region to 1 point if too large
        import warnings
        warnings.warn('Influent width was larger than channel width. Influent has been reset to one point.',UserWarning)
        in_out_points = 1

    for bb in np.arange(baffle_pairs):
        boundary[1:round(nx * baffle_length), round(int(2 * bb + 1) * ny * 1 / (2 * baffle_pairs + 1)) - 1] = 1
        psi[1:round(nx * baffle_length), round(int(2 * bb + 1) * ny * 1 / (2 * baffle_pairs + 1)) - 1] = flux
        boundary[round(nx * (1 - baffle_length) + 1):nx,
        round(int(2 * (bb + 1) * ny * 1 / (2 * baffle_pairs + 1))) - 1] = 1

    psi[0, 0:round(int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1))] = 0
    psi[-1, ny - round(int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1))):ny + 1] = flux

    for i in np.arange(in_out_points - 1):
        psi[0, round(
            int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - in_out_points / 2 + 1 + i) - 2)] = flux / in_out_points * (
                i + 1)
        psi[-1, ny - round(in_out_points / 2) - round(
            int(1 / 2 * ny * 1 / (2 * baffle_pairs + 1) - i))] = flux / in_out_points * (i + 1)

    #boundary[1:-1, 1:-1] *= 0
    #psi[1:-1, 1:-1] *= 0
    return boundary, psi

def save_velocity(ux,uy):
    global hrt, nx, ny, Lx, Ly, baffle_pairs, baffle_length
    import os
    from pathlib import Path
    file_a = 'hrt' + str(hrt).replace('.', '_') + '_nx' + str(nx) + '_ny' + str(ny) + '_Lx' + str(Lx) + '_Ly' + str(
        Ly) + '_pairs' + str(baffle_pairs) + '_width' + str(np.round(baffle_length, decimals=1)).replace('.',
                                                                                                         '_') + '.csv'
    file_x = 'Ux_' + file_a
    file_y = 'Uy_' + file_a
    data_folder = Path(os.getcwd(), "Output", "Velocity")
    np.savetxt(data_folder / file_x, ux, fmt='%.18e', delimiter=',')
    np.savetxt(data_folder / file_y, uy, fmt='%.18e', delimiter=',')

nx = 100
ny = 100
Lx = Domain_Config.Reactor['x_length'] # length of x axis in system
Ly = Domain_Config.Reactor['y_length']
Lz = Domain_Config.Reactor['z_length']

hrt = Domain_Config.Retention['time']    #/factor # hydraulic retention time
if Domain_Config.Retention['units'] == 'd':
    TC = 24*60*60
elif Domain_Config.Retention['units'] == 'h':
    TC = 60*60
elif Domain_Config.Retention['units'] == 'm':
    TC = 60
hrt *= TC  # The hydraulic retention time converted into seconds

flux = (Lx * Ly) / hrt  # The "area flux" through the system

baffle_length =  0*91/ 100  ##### This is the fraction of the tank a baffle takes up in x
baffle_pairs = 0*1 #### Number of baffle pairs (RHS+LHS) = 1 pair.
nxy = nx * ny
nxy_one = (nx + 1) * (ny + 1)

dx = Lx / nx
dy = Ly / ny
x = np.linspace(0, Lx, nx + 1).T
y = np.linspace(0, Ly, ny + 1).T
[yy, xx] = np.meshgrid(np.linspace(dy / 2, Ly - dy / 2, ny), np.linspace(dx / 2, Lx - dx / 2, nx))

influent_width = Domain_Config.Reactor['influent_width']#0.1* Lx#14*dy#0.8 * Ly
boundary, psi = boundary_init(influent_width)


system = solver.domain(Lx, Ly, Lz, nx=nx, ny=ny)
psi, boundary, in_out_points, in_start, out_start = system.influent_effluent_regions(baffle_pairs, baffle_length,
                                                                                    influent_width, psi, boundary, flux)

psi, ux, uy, resid, t = solver.steady_state(boundary, psi, nx, ny, dx, dy,error=1e-7,dt_min=4e-3)



plot_streamlines(xx,yy,ux,uy,title='Velocity Streamlines for a HRT of {}'.format(Domain_Config.Retention['time']) + ' ' + Domain_Config.Retention['units'])

plt.show()