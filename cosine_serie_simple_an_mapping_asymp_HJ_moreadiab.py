#!/scratch/mondelo/miniconda3/bin/python3
# coding: utf-8
import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit
import sys
import argparse
import os
import math
import subprocess
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator

def figsize(scale):
    #fig_width_pt = 422.52348  # Get this from LaTeX using \the\textwidth
    #fig_width_pt = 432.48181  # Get this from LaTeX using \the\textwidth
    fig_width_pt = 290.74263  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0  # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt*scale  # width in inches
    fig_height = fig_width*golden_mean  # height in inches
    #fig_height = fig_width*0.5  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {  # setup matplotlib to use latex for output
#    "pgf.texsystem": "pdflatex",  # change this if using xetex or luatex
    "text.usetex": False,  # use LaTeX to write all text
    "font.family": "sans-serif",
#    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": "Arial",
#    "font.monospace": [],
    "axes.labelsize": 24,  # LaTeX default is 10pt font.
    "axes.titlesize": 24,
    "font.size": 16,
    "legend.fontsize": 16,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.figsize": figsize(1.1378),  # default fig size of 0.9 textwidth
#    "pgf.preamble": "\\usepackage{siunitx}",
    "hatch.linewidth": 0.25
}
mpl.rcParams.update(pgf_with_latex)



# def FS(x, De, R0):
#     return De*((R0/x)**12-2*(R0/x)**6)
    
def FS(c0,c1,c2,c3,c4,c5,theta):
    fs = c0*math.cos(theta)+c1*math.cos(theta)+c2*math.cos(theta)+c3*math.cos(theta)+c4*math.cos(theta)+c5*math.cos(theta)
def ddx_FS(c0,c1,c2,c3,c4,c5,theta):
    dd_fs = c0*math.sin(theta)+c1*math.sin(theta)+c2*math.sin(theta)+c3*math.sin(theta)+c4*math.sin(theta)+c5*math.sin(theta)

def biexp(x, a, b, c, d):
    return a*np.exp(-x/b)+c*np.exp(-x/d)

def mod_exp(x, a, b):
    return np.exp(a*(x-b))

path=os.getcwd()

parser = argparse.ArgumentParser(description=
       "Performs an approximate transformation from diabatic to adiabatic "\
       "representation of a MCTDH propagation using an LVC Hamiltonian")
parser.add_argument("calcfolder")

# Initialize data structures
# --------------------------
colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
ev2Hartree = 1/27.2114
J2hartree=4.35974e18
angs2bohr = 1.88973 
displ=0.#25
M=2
N=4
N_coarse=4
E_off='nowinborhs_oldscan_HJ_Nc{}_mod'.format(N_coarse)

# Read functions from file and
# initialize data structures
# ------------------------------
raw_scan=np.loadtxt('scan_pes.dat')
#raw_scan=np.loadtxt('fullscan_corrected_swap.dat')

raw_scan_t=np.transpose(raw_scan)
ev_scan = np.copy(raw_scan_t)
ev_scan[1]= ev_scan[1]/ev2Hartree
vg0_min = np.min(ev_scan[1])
ev_scan[1]-= vg0_min 
ev_scan[2]+= ev_scan[1]
ev_scan[3]+= ev_scan[1]
ev_scan[0][22:]-=360


# raw_scan[:,1:]-=raw_scan[-1,1]
scan=ev_scan.copy()
points=scan[0]#*angs2bohr
#points_mod=points-displ
# scan[:,2]=(raw_scan[:,2]+raw_scan[:,3])/2+(raw_scan[:,2]-raw_scan[:,3])/8.5
# scan[:,3]=(raw_scan[:,2]+raw_scan[:,3])/2-(raw_scan[:,2]-raw_scan[:,3])/8.5
print(scan)
beta=np.zeros((scan.shape[1]))
diag=np.zeros((scan.shape[1]))
vg=np.zeros((scan.shape[1]))
ve=np.zeros((scan.shape[1]))

# Mass-freq, necessary for MFW coord representations
# --------------------------------------------------
mass_OT4=((4*12+32+2)*4+2)*1836
mass_inter=mass_OT4*mass_OT4/(mass_OT4+mass_OT4)

mass_OT=(4*12+32+2)*1836
mass_OT_chain=mass_OT*N_coarse+2*1836
mass_inter_coarse=mass_OT_chain/2


# Fit the scanned potentials to a FS form:
# ----------------------------------------
FS_lambd0, pcov_lambd0 = curve_fit(FS, points[1:], scan[1:,1])
np.savetxt('lambda0_fit_Eoff{}.dat'.format(E_off),FS_lambd0, header=' # Lennard-Jones form: D*((-R0/R)^6-(R0/R)**12) \n D    R0 ',
           footer='# freq={}'.format(np.sqrt((FS_lambd0[0]*2*(6/FS_lambd0[1])**2)/mass_inter)))
FS_lambd1, pcov_lambd1 = curve_fit(FS, points[1:], scan[1:,2]-scan[-1,2])
np.savetxt('lambda1_fit_Eoff{}.dat'.format(E_off),FS_lambd1, header=' # Lennard-Jones form: D*((-R0/R)^6-(R0/R)**12) \n D    R0 ',
           footer='# freq={}'.format(np.sqrt((FS_lambd1[0]*2*(6/FS_lambd1[1])**2)/mass_inter)))
FS_lambd2, pcov_lambd2 = curve_fit(FS, points[1:], scan[1:,3]-scan[-1,3])
np.savetxt('lambda2_fit_Eoff{}.dat'.format(E_off),FS_lambd2, header=' # Lennard-Jones form: D*((-R0/R)^6-(R0/R)**12) \n D    R0 ',
           footer='# freq={}'.format(np.sqrt((FS_lambd2[0]*2*(6/FS_lambd2[1])**2)/mass_inter)))


# Obtain the point-wise dependent
# values of beta and diag
# -------------------------------
dinf=scan[-1,2]#-(scan[-1,2]-scan[-1,3])/(np.cos(2*np.pi/3)-np.cos(1*np.pi/3))*np.cos(2*np.pi/3)
evals_reconstr=np.zeros((points.size, 9))
for i, point in enumerate(points):
    #beta[i]=(scan[i,2]-scan[i,3])/2/(np.cos(2*np.pi/3)-np.cos(1*np.pi/3))
    beta[i]=(FS(point-displ, *FS_lambd1)-FS(point-displ,*FS_lambd2))/2/(np.cos(2*np.pi/3)-np.cos(1*np.pi/3))
    #diag[i]=scan[i,2]-(scan[i,2]-scan[i,3])/(np.cos(2*np.pi/3)-np.cos(1*np.pi/3))*np.cos(2*np.pi/3)-dinf
    diag[i]=FS(point-displ, *FS_lambd1)+scan[-1,2]-(FS(point-displ, *FS_lambd1)-FS(point-displ, *FS_lambd2))/(np.cos(2*np.pi/3)-np.cos(1*np.pi/3))*np.cos(2*np.pi/3)-dinf
    #diag[i]=0.5*((scan[i,2]+scan[i,3])
    #             -((np.cos(2*np.pi/3)+np.cos(1*np.pi/3))/(np.cos(2*np.pi/3)-np.cos(1*np.pi/3)))
    #             *(scan[i,2]-scan[i,3])-2*scan[-1,1])
    vg[i]=(scan[i,1]-scan[-1,1])/(M*N)
    ve[i]=diag[i]-(M*N-1)*vg[i]
    #Sanity check: diagonalizing the resulting matrix
    mat=np.zeros((9,9))
    #mat+=np.eye(9)*(diag[i]+dinf)
    mat+=np.eye(9)*(ve[i]+(M*N-1)*vg[i]+dinf)
    for j in range(8):
        mat[j,j+1]=beta[i]
        mat[j+1,j]=beta[i]
    print(mat)
    evals_reconstr[i,:]=np.linalg.eigvalsh(mat)
    #print(evals[0], evals[1])
np.savetxt('Recontst_evals.txt', np.column_stack((points.T/angs2bohr,evals_reconstr)), delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')
FSreconstr=[]
for i in range(9):
    FSfit, pcov=curve_fit(FS, points[1:], evals_reconstr[1:,i]-evals_reconstr[-1,i])
    FSreconstr.append(FSfit)
print(FSreconstr)

# Fit diag and beta to a Lennard-jones function
# ---------------------------------------------
FS_diag, pcovdiag = curve_fit(FS, points[1:], diag[1:])
with open('dinf.dat','w') as out:
    for i,point in enumerate(points):
        out.write('{}  {} {}\n'.format(point, diag[i], FS(point, *FS_diag)))
FS_G, pcovG = curve_fit(FS, points[1:], vg[1:])
#print(FS_G)
FS_E, pcovE = curve_fit(FS, points[1:], ve[1:])
#print(FS_E)
beta_coef, pcov_beta=curve_fit(biexp, points[1:-1], beta[1:-1])
#print(beta_coef)
#Rmin_gs=pow(2,1/6)*FS_G[1]
beta_mings=biexp(FS_G[1],*beta_coef)/ev2Hartree

k_inter_gs=FS_G[0]*2*(6/FS_G[1])**2
freq_inter_gs=np.sqrt(k_inter_gs/mass_inter)

k_inter_es=FS_E[0]*2*(6/FS_E[1])**2
freq_inter_es=np.sqrt(k_inter_es/mass_inter)

# Plotting
# --------

fig = plt.figure()
ax = fig.add_subplot(111)

#ax.plot(points, (scan[:,1]-scan[-1,1])/ev2Hartree, 'x-')
#ax.plot(points, (scan[:,2]-scan[-1,1])/ev2Hartree,'x-')
#ax.plot(points, (scan[:,3]-scan[-1,1])/ev2Hartree,'x-')

finepoints=np.linspace(1,20, 150)
color=next(ax._get_lines.prop_cycler)['color']
ax.plot(points, scan[:,1]/ev2Hartree,'x',color=color)
ax.plot(finepoints, FS(finepoints,*FS_lambd0)/ev2Hartree, color=color, label=r'$\lambda_0$')
color=next(ax._get_lines.prop_cycler)['color']
ax.plot(points, scan[:,2]/ev2Hartree,'x',color=color)
ax.plot(finepoints, (FS(finepoints,*FS_lambd1)+scan[-1,2])/ev2Hartree,color=color, label=r'$\lambda_1$')
color=next(ax._get_lines.prop_cycler)['color']
ax.plot(points, scan[:,3]/ev2Hartree,'x', color=color)
ax.plot(finepoints, (FS(finepoints,*FS_lambd2)+scan[-1,2])/ev2Hartree,color=color, label=r'$\lambda_2$')

ax.set_xlim(1, 20)
ax.set_ylim(-1.5, 15)

ax.set_xlabel(r'R($a_0$)')
ax.set_ylabel(r'E (eV)')
ax.legend(loc=0, ncol=1,
         frameon=False,
         )
plt.tight_layout()
fig.savefig('Hagg_adiabpots_Eoff{}.pdf'.format(E_off))

plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)

#ax.plot(points, (scan[:,1]-scan[-1,1])/ev2Hartree, 'x-')
#ax.plot(points, (scan[:,2]-scan[-1,1])/ev2Hartree,'x-')
#ax.plot(points, (scan[:,3]-scan[-1,1])/ev2Hartree,'x-')
color=next(ax._get_lines.prop_cycler)['color']
ax.plot(points, vg/ev2Hartree,'x',color=color)
ax.plot(finepoints, FS(finepoints,*FS_G)/ev2Hartree, color=color, label=r'$v_g$')
color=next(ax._get_lines.prop_cycler)['color']
ax.plot(points, ve/ev2Hartree,'x',color=color)
ax.plot(finepoints, FS(finepoints,*FS_E)/ev2Hartree,color=color, label=r'$v_e$')
color=next(ax._get_lines.prop_cycler)['color']
ax.plot(points, beta/ev2Hartree,'x', color=color)
ax.plot(finepoints, biexp(finepoints,*beta_coef)/ev2Hartree,color=color, label=r'$\beta$')
#ax.scatter(Rmin_gs, FS(Rmin_gs,*FS_G)/ev2Hartree, color='red', marker='x')
ax.scatter(FS_G[1], FS(FS_G[1],*FS_G)/ev2Hartree, color='red', marker='o')

ax.set_xlim(0, 20)
ax.set_ylim(-1.5, 1)

ax.set_xlabel(r'R($a_0$)')
ax.set_ylabel(r'E (eV)')
ax.legend(loc=0, ncol=1,
         frameon=False,
         )
plt.tight_layout()
fig.savefig('Hagg_anmap_Eoff{}.pdf'.format(E_off))

figmfw = plt.figure()
axmfw = figmfw.add_subplot(111)

# Barrier harmonic potential
# --------------------------
h_barrier_R=50
h_barrier_a=ddx_FS(h_barrier_R, FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs))/2/h_barrier_R
h_barrier_b=FS(h_barrier_R, FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs))-h_barrier_a*h_barrier_R**2
e_barrier_b=FS(h_barrier_R, FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs))/ddx_FS(h_barrier_R, FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs))
print(e_barrier_b)
e_barrier_a=ddx_FS(h_barrier_R, FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs))*e_barrier_b*np.exp(-h_barrier_R/e_barrier_b)
print(e_barrier_a)
mfwfinepoints=finepoints/np.sqrt(mass_inter*freq_inter_gs)
mfwfinepoints=np.linspace(1,150,200)

#barrier_coef, pcov_barrier=curve_fit(biexp, mfwfinepoints[55:],-e_barrier_a*np.exp((2*h_barrier_R-mfwfinepoints[55:])/e_barrier_b)
#                                                          +2*FS(h_barrier_R,FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs)),
#                                     p0=(1,-1,1,-1))

color=next(axmfw._get_lines.prop_cycler)['color']
axmfw.plot(points*np.sqrt(mass_inter*freq_inter_gs), vg/ev2Hartree,'x',color=color)
axmfw.plot(mfwfinepoints, FS(mfwfinepoints,FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs))/ev2Hartree, color=color, label=r'$v_g$')
##axmfw.plot(mfwfinepoints, (h_barrier_a*mfwfinepoints**2+h_barrier_b)/ev2Hartree, color='red', linestyle=':')
###axmfw.plot(mfwfinepoints, (e_barrier_a*np.exp(mfwfinepoints/e_barrier_b))/ev2Hartree, color='black',linestyle=':')
##axmfw.plot(mfwfinepoints, (-e_barrier_a*np.exp((2*h_barrier_R-mfwfinepoints)/e_barrier_b)
##                            +2*FS(h_barrier_R,FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs)))/ev2Hartree, color='black',linestyle=':')
##axmfw.plot(mfwfinepoints, biexp(mfwfinepoints,barrier_coef[0], barrier_coef[1], barrier_coef[2], barrier_coef[3])/ev2Hartree, color='black',linestyle=':')
###axmfw.plot(mfwfinepoints, mod_exp(mfwfinepoints,barrier_coef[0], barrier_coef[1]*np.sqrt(mass_inter*freq_inter_gs), barrier_coef[2], barrier_coef[3]*np.sqrt(mass_inter*freq_inter_gs))/ev2Hartree, color='black',linestyle=':')
###axmfw.plot(mfwfinepoints, (FS(mfwfinepoints,FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs))
###    +FS(2*FS_G[1]*np.sqrt(mass_inter*freq_inter_gs)-mfwfinepoints,FS_G[0], FS_G[1]*np.sqrt(mass_inter*freq_inter_gs)))/2/ev2Hartree, color=color, label=r'$v_g$', linestyle=':')
#axmfw.plot(mfwfinepoints, (FS_G[0]*2*(6/(FS_G[1]*np.sqrt(mass_inter*freq_inter_gs)))**2)*(mfwfinepoints-FS_G[1]*np.sqrt(mass_inter*freq_inter_gs))**2/ev2Hartree-FS_G[0]/ev2Hartree, color='red', label=r'$v_g$')
color=next(axmfw._get_lines.prop_cycler)['color']
axmfw.plot(points*np.sqrt(mass_inter*freq_inter_gs), ve/ev2Hartree,'x',color=color)
axmfw.plot(mfwfinepoints, FS(mfwfinepoints,FS_E[0], FS_E[1]*np.sqrt(mass_inter*freq_inter_gs))/ev2Hartree,color=color, label=r'$v_e$')
#axmfw.plot(mfwfinepoints, (FS(mfwfinepoints,FS_E[0], FS_E[1]*np.sqrt(mass_inter*freq_inter_gs))
#    +FS(2*FS_G[1]*np.sqrt(mass_inter*freq_inter_gs)-mfwfinepoints,FS_E[0], FS_E[1]*np.sqrt(mass_inter*freq_inter_gs)))/2/ev2Hartree, color=color, label=r'$v_e$', linestyle=':')
color=next(axmfw._get_lines.prop_cycler)['color']
#axmfw.plot(points/np.sqrt(mass_inter*freq_inter_gs), beta/ev2Hartree,'o', color=color)
axmfw.plot(mfwfinepoints, biexp(mfwfinepoints,beta_coef[0], beta_coef[1]*np.sqrt(mass_inter*freq_inter_gs), beta_coef[2], beta_coef[3]*np.sqrt(mass_inter*freq_inter_gs))/ev2Hartree,color=color, label=r'$\beta$')
#axmfw.plot(mfwfinepoints, (biexp(mfwfinepoints,
#                                beta_coef[0], beta_coef[1]*np.sqrt(mass_inter*freq_inter_gs),
#                                beta[2], beta[3]*np.sqrt(mass_inter*freq_inter_gs))
#                           +biexp(2*FS_G[1]*np.sqrt(mass_inter*freq_inter_gs)-mfwfinepoints,
#                                beta_coef[0], beta_coef[1]*np.sqrt(mass_inter*freq_inter_gs),
#                                beta[2], beta[3]*np.sqrt(mass_inter*freq_inter_gs))
#                           )/2/ev2Hartree,color=color, label=r'$\beta$', linestyle=':')
#ax.scatter(Rmin_gs, FS(Rmin_gs,*FS_G)/ev2Hartree, color='red', marker='x')
#axmfw.scatter(FS_G[1], FS(FS_G[1],*FS_G)/ev2Hartree, color='red', marker='o')

axmfw.set_xlim(30, 150)
axmfw.set_ylim(-1.5, 1)

axmfw.set_xlabel(r'R($a_0$)/$\omega m$')
axmfw.set_ylabel(r'E (??)')
axmfw.legend(loc=0, ncol=1,
         frameon=False,
         )
plt.tight_layout()
figmfw.savefig('Hagg_anmap_mfw_Eoff{}.pdf'.format(E_off))



np.savetxt('beta_fit_Eoff{}.dat'.format(E_off),beta_coef, header=' # Exponential form: a*exp(-x/b)+c*(-x/d) \n a    b    c   d')
np.savetxt('vg_fit_Eoff{}.dat'.format(E_off),FS_G, header=' # Lennard-Jones form: D*((-R0/R)^6-(R0/R)**12) \n D    R0 ',
           footer='# freq={}'.format(np.sqrt((FS_G[0]*2*(6/FS_G[1])**2)/mass_inter)))
#np.savetxt('vg_fit_expcorr_Eoff{}.dat'.format(E_off),barrier_coef, header=' # Exponential form: a*exp(-x/b)+c*(-x/d) \n a    b    c   d')
np.savetxt('ve_fit_Eoff{}.dat'.format(E_off),FS_E, header=' # Lennard-Jones form: D*((-R0/R)^6-(R0/R)**12) \n D    R0 ',
           footer='# freq={}'.format(np.sqrt((FS_E[0]*2*(6/FS_E[1])**2)/mass_inter)))
np.savetxt('diag_fit_Eoff{}.dat'.format(E_off),FS_diag, header=' # Lennard-Jones form: D*((-R0/R)^6-(R0/R)**12) \n D    R0 ')

with open('beta_fit_Eoff{}.dat'.format(E_off),'a') as out:
    out.write('Value of beta at GS equilibrium position: {} eV'.format(beta_mings))
with open('vg_fit_Eoff{}.dat'.format(E_off),'a') as out:
    out.write('Value of a for parabolic barrier: {} Eh'.format(h_barrier_a))
    out.write('Value of b for parabolic barrier: {} Eh'.format(h_barrier_b))

print(mass_inter, freq_inter_gs)
print(beta_coef[0], beta_coef[1]*np.sqrt(mass_inter*freq_inter_gs), beta_coef[2], beta_coef[3]*np.sqrt(mass_inter*freq_inter_gs))

# Sanity check: adiabatic potentials after correction
print('Sanity Check')
lambda1_corr=[]
lambda2_corr=[]
for i, point in enumerate(points[:-1]):
    mat=np.zeros((2,2))
    mat+=np.eye(2)*(vg[i]+ve[i])
    #if point>h_barrier_R/np.sqrt(mass_inter*freq_inter_gs):
    #    mat+=np.eye(2)*biexp(point, barrier_coef[0], barrier_coef[1]/np.sqrt(mass_inter*freq_inter_gs), barrier_coef[2], barrier_coef[3]/np.sqrt(mass_inter*freq_inter_gs))
    mat[0,1]=beta[i]
    mat[1,0]=beta[i]
    print(point)
    print(mat)
    evals=np.linalg.eigvals(mat)
    lambda1_corr.append(evals[0])
    lambda2_corr.append(evals[1])

lambda1_corr=np.array(lambda1_corr)
lambda2_corr=np.array(lambda2_corr)
lambda1=[]
lambda2=[]
for i, point in enumerate(points[:-1]):
    mat=np.zeros((2,2))
    mat+=np.eye(2)*(vg[i]+ve[i])
    mat[0,1]=beta[i]
    mat[1,0]=beta[i]
    evals=np.linalg.eigvals(mat)
    lambda1.append(evals[0])
    lambda2.append(evals[1])

lambda1=np.array(lambda1)
lambda2=np.array(lambda2)

plt.close()
fig=plt.figure()
ax=fig.add_subplot(111)

ax.plot(points[:-1], lambda1/ev2Hartree, color=color)
ax.plot(points[:-1], lambda1_corr/ev2Hartree, color=color, linestyle=":")
color=next(axmfw._get_lines.prop_cycler)['color']
ax.plot(points[:-1], lambda2/ev2Hartree, color=color)
ax.plot(points[:-1], lambda2_corr/ev2Hartree, color=color, linestyle=":")
ax.set_ylim(-1.5,1)
ax.set_xlim(0,20)
ax.set_xlabel(r'R($a_0$)')
ax.set_ylabel(r'E (eV)')
plt.tight_layout()

fig.savefig('Adiabatic_potentials_corrected.pdf')
plt.close()

#Now find the coarse-grained potentials for N_coarse number of monomers per chain
vg_coarse=np.zeros((scan.shape[0]))
ve_coarse=np.zeros((scan.shape[0]))
for i, point in enumerate(points):
    vg_coarse[i]=N_coarse*vg[i]
    ve_coarse[i]=diag[i]-(M-1)*vg_coarse[i]

FS_G_coarse, pcovG = curve_fit(FS, points[1:], vg_coarse[1:])
np.savetxt('vgCG_N{}_fit_Eoff{}.dat'.format(N_coarse,E_off),FS_G_coarse, header=' # Lennard-Jones form: D*((-R0/R)^6-(R0/R)**12) \n D    R0 ',
           footer='# freq={}'.format(np.sqrt((FS_G_coarse[0]*2*(6/FS_G_coarse[1])**2)/mass_inter_coarse)))
freq_gs_coarse=np.sqrt((FS_G_coarse[0]*2*(6/FS_G_coarse[1])**2)/mass_inter_coarse)
with open('vgcoarse_N{}_mfw.dat'.format(N_coarse),'w') as out:
    for i,point in enumerate(points):
        out.write('{}  {}\n'.format(point*np.sqrt(mass_inter_coarse*freq_gs_coarse), FS(point*np.sqrt(mass_inter_coarse*freq_gs_coarse),FS_G_coarse[0], FS_G_coarse[1]*np.sqrt(mass_inter_coarse*freq_gs_coarse))))
with open('vgcoarse_N{}.dat'.format(N_coarse),'w') as out:
    for i,point in enumerate(finepoints):
        out.write('{}  {} \n'.format(point, FS(point, *FS_G_coarse)))
#mask=[False if i==5 or i==3 or i==1 or i==0 else True for i in range(ve_coarse.size)]
mask=[False if i==1 or i==1 or i==0 else True for i in range(ve_coarse.size)]
#mask=[False if i==7 or i==8 or i==2 or i==3 or i==5 or i==4 or i==6 or i==9 or i==10 or i==11 else True for i in range(ve_coarse.size)]
#mask=[False if i in range(4,12) or i==0 else True for i in range(ve_coarse.size)]
#print(mask)
FS_E_coarse, pcovG = curve_fit(FS, points[mask], ve_coarse[mask])
with open('vecoarse_N{}.dat'.format(N_coarse),'w') as out:
    for i,point in enumerate(finepoints):
        out.write('{}  {} \n'.format(point, FS(point,*FS_E_coarse)))
np.savetxt('veCG_N{}_fit_Eoff{}.dat'.format(N_coarse,E_off),FS_E_coarse, header=' # Lennard-Jones form: D*((-R0/R)^6-(R0/R)**12) \n D    R0 ',
           footer='# freq={}'.format(np.sqrt((FS_G[0]*2*(6/FS_E_coarse[1])**2)/mass_inter_coarse)))

with open('beta.dat','w') as out:
    for i,point in enumerate(finepoints):
        out.write('{}  {} \n'.format(point, biexp(point,*beta_coef)))

# Plot of final figure for paper
#===============================
figwidth=650*2#px
ratio=1.33
numPanels = 2

lm=110#px#105#px
tm=50#px
bm=92#px

spacer=100#px
bigSpacer=10#px
legendSpacer = 13#px
vertSpacer = 75#px

legendPad = 7#px
legendWidth = 75#px
cbar_width = 12#px

panelWidth = (figwidth-lm-2*legendPad-2*legendWidth-vertSpacer)/2#360#px
panelHeight = panelWidth/ratio#px

figheight=bm+tm+numPanels*panelHeight+(numPanels-1)*spacer+bigSpacer#px

lm /= figwidth
bm /= figheight
panelWidth /= figwidth
panelHeight /= figheight
spacer /= figheight
bigSpacer /= figheight
legendSpacer /= figheight
vertSpacer /= figwidth
legendPad /= figwidth
legendWidth /= figwidth
cbar_width /= figwidth


figwidth /= 100
figheight /=100


print('DeGm=',FS_G[0]/ev2Hartree)
print('DeEm=',FS_E[0]/ev2Hartree)
print('sigmaGm=',FS_G[1]/angs2bohr)
print('simgaEm=',FS_E[1]/angs2bohr)
print('betaa=',beta_coef[0]/ev2Hartree)
print('betab=',beta_coef[1]/angs2bohr)
print('betac=',beta_coef[2]/ev2Hartree)
print('betad=',beta_coef[3]/angs2bohr)

print('freqinterGm=',freq_inter_gs/ev2Hartree)
print('freqinterEm=',freq_inter_es/ev2Hartree)

k_inter_ol_gs=FS_G_coarse[0]*2*(6/FS_G_coarse[1])**2
freq_inter_gs_olig=np.sqrt(k_inter_ol_gs/mass_inter)/ev2Hartree
print('freqinterGc=',freq_inter_gs_olig)
k_inter_ol_es=FS_E_coarse[0]*2*(6/FS_E_coarse[1])**2
freq_inter_es_olig=np.sqrt(k_inter_ol_es/mass_inter)/ev2Hartree
print('freqinterEc=',freq_inter_es_olig)

print('DeGc=',FS_G_coarse[0]/ev2Hartree)
print('DeEc=',FS_E_coarse[0]/ev2Hartree)
print('simgaGc=',FS_G_coarse[1]/angs2bohr)
print('sigmaEc=',FS_E_coarse[1]/angs2bohr)

#FS_lambd0[0]/=J2hartree
#FS_lambd1[0]/=J2hartree
#FS_lambd2[0]/=J2hartree
#FS_lambd0[1]/=angs2bohr/1e10
#FS_lambd1[1]/=angs2bohr/1e10
#FS_lambd2[1]/=angs2bohr/1e10

finepoints=np.linspace(1,20, 150)#/angs2bohr*1e-10

fig_paper = plt.figure(figsize=(figwidth, figheight))
ax_adiab = fig_paper.add_axes([lm, bm+(numPanels-1)*(panelHeight+spacer)+bigSpacer, panelWidth, panelHeight])
ax_radia = fig_paper.add_axes([lm+panelWidth+legendPad+legendWidth+vertSpacer, bm+(numPanels-1)*(panelHeight+spacer)+bigSpacer, panelWidth, panelHeight])
ax_monom = fig_paper.add_axes([lm, bm+(numPanels-2)*(panelHeight+spacer)+bigSpacer, panelWidth, panelHeight])
ax_oligo = fig_paper.add_axes([lm+panelWidth+legendPad+legendWidth+vertSpacer, bm+(numPanels-2)*(panelHeight+spacer)+bigSpacer, panelWidth, panelHeight])

ax_adiab.plot(finepoints/angs2bohr, FS(finepoints, *FS_lambd0)/ev2Hartree, color='k', label=r'$S_0^{(2)}$')
#ax_adiab.scatter(points[1:]/angs2bohr, scan[1:,1]/ev2Hartree, color=colorcycle[0], marker='o')
#ax_adiab.scatter(points[1:]/angs2bohr, evals_reconstr[1:,0]/ev2Hartree, color=colorcycle[0], marker='o')
#ax_adiab.scatter(points[1:]/angs2bohr, evals_reconstr[1:,8]/ev2Hartree, color=colorcycle[0], marker='o')
ax_adiab.plot(finepoints/angs2bohr, (FS(finepoints, *FS_lambd1)+scan[-1,2])/ev2Hartree, color=colorcycle[3], label=r'$S_1^{(2)}$')
#ax_adiab.scatter(points[1:]/angs2bohr, scan[1:,2]/ev2Hartree, color=colorcycle[1], marker='o')
ax_adiab.plot(finepoints/angs2bohr, (FS(finepoints, *FS_lambd2)+scan[-1,2])/ev2Hartree, color=colorcycle[4], label=r'$S_2^{(2)}$')
#ax_adiab.scatter(points[1:]/angs2bohr, scan[1:,3]/ev2Hartree, color=colorcycle[2], marker='o')

ax_adiab.set_xlim(5/angs2bohr,14/angs2bohr)
ax_adiab.set_ylim(-2,5)
ax_adiab.legend(loc=0)

ax_adiab.set_xlabel(r'$R_{\gamma\!,\!\gamma\!+\!\mathregular{1}}$ / $\AA$')
ax_adiab.set_ylabel(r'E / eV')

index=1
linest='-'
for i in range(9):
    if index>4:
        index=1
        linest='--'
    if i==8: linest=':'
    linecolor=colorcycle[index+2]
    ax_radia.plot(finepoints/angs2bohr, FS(finepoints, *FSreconstr[i])/ev2Hartree+evals_reconstr[-1,i]/ev2Hartree, label=r'$S_{}^{{(9)}}$'.format(i+1), color=linecolor, linestyle=linest)
    #ax_radia.scatter(points[1:]/angs2bohr, evals_reconstr[1:,i]/ev2Hartree, marker='o')
    index+=1
ax_radia.set_xlim(5/angs2bohr,14/angs2bohr)
ax_radia.set_ylim(2,3.5)
ax_radia.yaxis.set_major_locator(MaxNLocator(4)) 
ax_radia.legend(loc=0, ncol=1, fontsize=14)

ax_radia.set_xlabel(r'$R_{\gamma\!,\!\gamma\!+\!\mathregular{1}}$ / $\AA$')
ax_radia.set_ylabel(r'E / eV')

ax_monom.plot(finepoints/angs2bohr, FS(finepoints, *FS_G)/ev2Hartree, color=colorcycle[0], label=r'$v^{HJ}_G$')
#ax_monom.scatter(points[1:]/angs2bohr, vg[1:]/ev2Hartree, color=colorcycle[3], marker='o')
ax_monom.plot(finepoints/angs2bohr, FS(finepoints, *FS_E)/ev2Hartree, color=colorcycle[1], label=r'$v^{HJ}_E$')
#ax_monom.scatter(points[1:]/angs2bohr, ve[1:]/ev2Hartree, color=colorcycle[4], marker='o')
ax_monom.plot(finepoints/angs2bohr, biexp(finepoints, *beta_coef)/ev2Hartree, color=colorcycle[2], label=r'$\beta$')
#ax_monom.scatter(points[1:]/angs2bohr, beta[1:]/ev2Hartree, color=colorcycle[5], marker='o')
ax_monom.yaxis.set_major_locator(MaxNLocator(5)) 

ax_monom.set_xlim(3/angs2bohr,14/angs2bohr)
ax_monom.set_ylim(-2,2)

ax_monom.set_xlabel(r'$R_{\gamma\!,\!\gamma\!+\!\mathregular{1}}$ / $\AA$')
ax_monom.set_ylabel(r'E / eV')
ax_monom.legend(loc=0)

ax_oligo.plot(finepoints/angs2bohr, FS(finepoints, *FS_G_coarse)/ev2Hartree, color=colorcycle[0], label=r'$v^{H}_G$')
#ax_oligo.scatter(points[1:]/angs2bohr, vg_coarse[1:]/ev2Hartree, color=colorcycle[3], marker='o')
ax_oligo.plot(finepoints/angs2bohr, FS(finepoints, *FS_E_coarse)/ev2Hartree, color=colorcycle[1], label=r'$v^{H}_E$')
#ax_oligo.scatter(points[1:]/angs2bohr, ve_coarse[1:]/ev2Hartree, color=colorcycle[4], marker='o')
ax_oligo.plot(finepoints/angs2bohr, biexp(finepoints, *beta_coef)/ev2Hartree, color=colorcycle[2], label=r'$\beta$')
#ax_oligo.scatter(points[1:]/angs2bohr, beta[1:]/ev2Hartree, color=colorcycle[5], marker='o')
ax_oligo.yaxis.set_major_locator(MaxNLocator(5)) 

ax_oligo.set_xlim(3/angs2bohr,14/angs2bohr)
ax_oligo.set_ylim(-2,2)

ax_oligo.set_xlabel(r'$R_{\gamma\!,\!\gamma\!+\!\mathregular{1}}$ / $\AA$')
ax_oligo.set_ylabel(r'E / eV')
ax_oligo.legend(loc=0)

trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
ax_adiab.text(-0.1, 0.9, '(a)', transform=ax_adiab.transAxes + trans,
        fontsize=28, va='bottom')
ax_radia.text(-0.1, 0.9, '(b)', transform=ax_radia.transAxes + trans,
        fontsize=28, va='bottom')
ax_monom.text(-0.1, 0.9, '(c)', transform=ax_monom.transAxes + trans,
        fontsize=28, va='bottom')
ax_oligo.text(-0.1, 0.9, '(d)', transform=ax_oligo.transAxes + trans,
        fontsize=28, va='bottom')

#ax_adiab.text(0.5, 1.0, 'M=2', transform=ax_adiab.transAxes + trans,
#        fontsize=28, va='bottom')
#
#ax_radia.text(0.5, 1.0, 'M=8', transform=ax_radia.transAxes + trans,
#        fontsize=28, va='bottom')

fig_paper.savefig("Adiab_vs_monomer_pots_N{}.pdf".format(N_coarse))
