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

    channel_width = Lx / (baffle_pairs * 2 + 1)
    if channel_width > influent_width:
        in_out_points = influent_width / dx
    else:  # Default influent region to 1 point if too large
        import warnings
        warnings.warn('Influent width was larger than channel width. Influent has been reset to one point.',UserWarning)
        in_out_points = 1

    psi[:,0] = 0
    psi[:,-1] = flux
    psi[0,:] = flux
    psi[-1, -1] = 0
    psi[-1, :] = 0

    for i in np.arange(in_out_points - 1):
        psi[int(i+1),0] = flux - flux / in_out_points * (i + 1)
        psi[int(nx-i-1),-1] = flux / in_out_points * (i + 1)
        #Boundary on same side
        psi[int(nx-14+i),0] = flux / in_out_points * (i + 1)
    #boundary on same side
    psi[-1, :] = flux
    psi[:,-1] = flux

    # Original backwards flow
    # psi[:, 0] = flux
    # psi[:, -1] = 0
    # psi[0, :] = 0
    # psi[-1, -1] = flux
    # psi[-1, :] = flux
    #
    # for i in np.arange(in_out_points - 1):
    #     psi[int(i+1),0] =  flux / in_out_points * (i + 1)
    #     psi[int(nx-i-1),-1] = flux - flux / in_out_points * (i + 1)

    in_start = int(0)
    out_start = int(nx - i - 1)
    in_out_points = int(in_out_points)

    return [psi, boundary, in_out_points, in_start, out_start]

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

circle = False # Change to true to include circular obstical
square = False
nx = 150
ny = 150
Lx = 0.09#0.32 # length of x axis in system
Ly = 0.09#0.1
Lz = 0.09
hrt = 1#/factor # hydraulic retention time
TC = (60)  # Time conversion, notation is consistent through other codes
hrt *= TC  # The hydraulic retention time converted into seconds

flux = (Lx * Ly) / hrt  # The "area flux" through the system

baffle_length =  0*91 / 100  ##### This is the fraction of the tank a baffle takes up in x
baffle_pairs = 0*5 #### Number of baffle pairs (RHS+LHS) = 1 pair.
nxy = nx * ny
nxy_one = (nx + 1) * (ny + 1)

dx = Lx / nx
dy = Ly / ny
x = np.linspace(0, Lx, nx + 1).T
y = np.linspace(0, Ly, ny + 1).T
[yy, xx] = np.meshgrid(np.linspace(dy / 2, Ly - dy / 2, ny), np.linspace(dx / 2, Lx - dx / 2, nx))

influent_width = 0.1* Lx
psi, boundary, in_out_points, in_start, out_start = boundary_init(influent_width)


system = solver.domain(Lx, Ly, Lz, nx=nx, ny=ny)
#psi, boundary, in_out_points, in_start, out_start = system.influent_effluent_regions(baffle_pairs, baffle_length,
                                                                                    # influent_width, psi, boundary, flux)


if circle:
    nx1 = nx+1
    ny1 = ny+1
    dx1 = Lx / nx1
    dy1 = Ly / ny1
    [yy1, xx1] = np.meshgrid(np.linspace(dy1 / 2, Ly - dy1 / 2, ny1), np.linspace(dx1 / 2, Lx - dx1 / 2, nx1))
    a = Lx/4
    b = Ly/2
    r = np.min((Lx,Ly))/12
    kx = nx1 / ny1
    ky = 1
    circle = ((xx1 - a)/2) ** 2 + ((yy1 - b)) ** 2
    donut = np.logical_and(circle < (r ** 2), circle >= 0)
    boundary[1:-2,1:-2] = donut[1:-2,1:-2]
    psi[1:-2, 1:-2] = donut[1:-2, 1:-2]*flux/2

if square:
    square_object_array = np.zeros(boundary.shape)
    square_object_array[int(nx / 4):int(nx / 4 + 20), 10:91] = 1
    square_object_array[int(nx / 4+ 60):int(nx / 4 + 100), 40:61] = 1
    square_object_array[int(nx / 4+ 100):int(nx / 4 +120), 10:91] = 1
    square_object_array[int(nx / 4 + 120):int(nx / 4 + 160), 40:61] = 1

    psi[square_object_array == 1] = flux / 2

    boundary[square_object_array == 1] = 1

    bdata = psi[boundary == 1]


psi, ux, uy, resid = solver.steady_state(boundary, psi, nx, ny, dx, dy,error=1e-7,dt_min=1e-1)

plot_streamlines(xx,yy,ux,uy,new_fig= True,title = 'Corner to Corner HRT: {} Min(s)'.format(hrt/TC))
ux_i = 0.5 * (ux[0:ux.shape[0] - 1, :] + ux[1:ux.shape[0], :])
uy_i = 0.5 * (uy[:, 0:uy.shape[1] - 1] + uy[:, 1:uy.shape[1]])
#save_figure('HRT_1_hour_influent_width_Ly02')
ux_vol_int = (((ux_i<0)*dx).sum(0)*dy).sum()
uy_vol_int = (((uy_i<0)*dy).sum(0)*dx).sum()
area = Lx*Ly
ux_est = ux_vol_int/area * 100
uy_est = uy_vol_int/area * 100
print('Estimated Area with recirculation / stagnation \n Ux Estimate:{:.2f} %\n Uy Estimate {:.2f} %\n '.format(ux_est,uy_est))

plt.figure()
plt.subplot(121)
plt.spy(ux_i<0)
plt.title('Ux < 0')
plt.subplot(122)                
plt.spy(uy_i<0)
plt.title('Uy < 0')

plt.show()
print(ux)                
                         