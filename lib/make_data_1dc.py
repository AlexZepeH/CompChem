#!/usr/bin/python3
"""
 Generation of training data for NN model of coupled multichromophore system.
 1D Chain System.

 Usage:
  [anything]

 Last revision:
   Frankfurt, 10 April 2022
"""
import sys
if len(sys.argv)==1:
   print(__doc__); exit()
import string, numpy, math, os, scipy, glob, datetime
import gefp, libbbg, psi4, oepdev
#import fortran_modules
#import neural, fragment
import local_lib
import scipy.optimize
numpy.random.seed(456)

NSTATES = 6
STATE_1 = 3
STATE_2 = 4
STATE_MON = 2
psi4.set_options({"scf_type"       : "direct", "cis_type": "davidson_liu",
                  "basis"          : "6-31G*", "DAVIDSON_LIU_NROOTS": NSTATES,
                  "guess"          : "gwh", "CIS_SCHWARTZ_CUTOFF": 1e-30,
                  "df_scf_guess"   : False, "freeze_core": "True",
                  "e_convergence"  : 1e-10    ,
                  "d_convergence"  : 1e-10    ,
                  "puream"         : False   ,
                  "print"          : 1       })

psi4.core.set_output_file(sys.argv[0].replace('.py','.log'), False)



class DataSymmetric:
  "Data chunk for symmetric dimer or one monomer"
  def __init__(self, v_0, e_ex, v_eet=None, label='untitled'):
      self.v_0  = v_0      # ground state energy
      self.e_ex = e_ex     # excitation energy (from ground state to excited state)
      self.v_eet= v_eet    # EET coupling constant
      self.label= label
  def __repr__(self):
      log = "%s\n"                    % self.label
      log+= "v_0   = %14.6E [a.u.]\n" % self.v_0
      log+= "e_ex  = %14.6E [a.u.]\n" % self.e_ex
      if self.v_eet is not None:
         log+= "v_eet = %14.6E [a.u.]\n" % self.v_eet
      return str(log)

#TODO: add ArgParser and nicer interface to the training to make it a general tool
class Samples: 
   """
 Creates samples for network training for coupled multichromophore system.

 Generates:
  * Gaussian input files for calculations of one-particle density matrices and adiabatic energies
  * Data files for training:
    - microdisplatements
    - inter-fragment coordinates
    - electrostatic potential on atoms due to other fragment
    - after executing Gaussian jobs (log files needed): 
      -- diabatic potential matrix elements v_00, v_11, v_22, v_12
      -- adiabatic energies E_0, E_1, E_2

 Notes:
  * assumes symmetric dimer displacements, so that v_11 = v_22.
    This facilitate the definition and the fitting of the diagonal
    potentials.
"""
   def __init__(self, log, directory='data', n_macro=10, n_micro=30, 
                                             mindist=3.5, mintransl=2.5, transl_span=3.0,
                                             s_micro=0.1, rot_amplitude=30.0, atid=False,
                                             selected_modes=None): #OK

       # read Gaussian log file
       mol = gefp.core.utilities.psi_molecule_from_gaussian_log(log)
       geom, mass, elem, elez, uniq = mol.to_arrays()

       charges = gefp.core.utilities.read_mulliken_charges_from_gaussian_log(log)
      #charges = numpy.array([-5.699415E-01,+5.699415E-01,-1.247683E-01,-1.247683E-01,+1.247683E-01,+1.247683E-01])
      #charges = numpy.array([+6.0,+6.0,+1.0,+1.0,+1.0,+1.0])
       lvec = gefp.core.utilities.read_wilson_matrix_from_gaussian_log(log) * libbbg.units.UNITS.BohrToAngstrom
       if selected_modes is not None:
          lvec = lvec[numpy.array(selected_modes, int)-1].copy()
     
       # options 
       self.n_macro = n_macro
       self.n_micro = n_micro
       self.s_micro = s_micro
       self.mindist = mindist
       self.mintransl= mintransl
       self.atid    = atid
       self.rot_amplitude = rot_amplitude
       self.transl_span = transl_span

       # fragment object
       self.frag_0 = local_lib.Fragment(xyz=geom*libbbg.units.UNITS.BohrToAngstrom, 
                                       lvec=lvec, atoms=elem, charges=charges, idx=self.atid,
                                 selected_modes=selected_modes)
       self.frag_0.move_to_origin()

       # data directory
       self.prefdir = './'
       self.directory = directory
       if directory is not None:
          if os.path.exists(directory): 
             assert os.path.isdir(directory), "The %s is not a directory!" % directory
          try:
         #os.system('rm -rv ./%s' % directory)
             os.system('mkdir -p ./%s' % directory)
          except: 
             print(" Warning: the data directory '%s' already exists." % directory)
          self.prefdir+='%s/' % directory

       # data for isolated fragment
       self.data_0 = self._compute_isolated_fragment(self.frag_0)
       psi4.core.print_out(str(self.data_0))

   def _compute_isolated_fragment(self, frag, state=STATE_MON): #OK
       "Compute ground state energy and excitation energy of isolated fragment"
       psi = frag.make_input_with(no_file=True, psi=True)
       mol = psi4.geometry(psi)

       # SCF
       v0, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)
       # CIS
       cis = oepdev.CISComputer.build("RESTRICTED", wfn, psi4.core.get_options(), "RHF")
       cis.compute()
       cis.clear_dpd()
       psi4.core.clean()
       e_ex = cis.eigenvalues().get(state-1)
       return DataSymmetric(v0, e_ex, label='==> Isolated Fragment Energetic Properties <==')

   def _compute_diabatic_pes(self, E_1, E_2, v_12): #OK
       DE = E_2 - E_1
       SE = E_2 + E_1
       k = v_12*v_12*4.0 / (DE*DE)
       if k > 1.0: k = 1.0
       d = math.sqrt(1.0 - k)
       v_11 = SE / 2.0 - DE / 2.0 * d
       v_22 = SE / 2.0 + DE / 2.0 * d
       return v_11, v_22

   # --- operational modes

   def make_inputs(self): #OK
       "Make the input files and part of training data"
       print("Making Gaussian input files...")

       # initialize the data files
       data_file_microdisplacements = open('data_micro.dat','w')
       data_file_interfragment      = open('data_inter.dat','w')
       data_file_electrostatic      = open('data_elect.dat','w')
       data_file_electrostatic_0    = open('data_elect_0.dat','w')
       data_file_energy             = open('data_energy.dat', 'w')

       # make displacements
       t0 = numpy.zeros(3)
       I = 1
       for macro_displacement in range(1,self.n_macro+1):
           frag_1 = self.frag_0.clone()
           frag_2 = self.frag_0.clone()
           while frag_1.minimum_distance(frag_2) < self.mindist:

               r, t = frag_1.generate_random_rottransl(transl_span=self.transl_span, rot_amplitude=self.rot_amplitude,
                                                       min_transl=self.mintransl, atid=self.atid)
               r1, r2 = frag_1.random_rotation_opposite_matrices(amplitude=self.rot_amplitude, random_axis=True); del r
              #r2[0,0] = -r2[0,0]; r2[1,1] = -r2[1,1]; r2[2,2] = -r2[2,2]
               r_inv = -numpy.identity(3)

              #frag_1.translate(t)
              #frag_1 = self.frag_0.make_macrodisplaced(rot_trans=(   r,t ))
              #frag_2 = self.frag_0.make_macrodisplaced(rot_trans=( r.T,t0))

               frag_1 = self.frag_0.clone()
               frag_2 = self.frag_0.clone(); #frag_2.rotate(r_inv)
               frag_1 = frag_1.make_translated(+0.5*t)
               frag_2 = frag_2.make_translated(-0.5*t)
               frag_1 = frag_1.make_macrodisplaced(rot_trans=( r1 ,t0))
               frag_2 = frag_2.make_macrodisplaced(rot_trans=( r2 ,t0))

           frag_1_nomicro = frag_1.clone()
           frag_2_nomicro = frag_2.clone()

           # input for reference x=0
           label_nomicro = "_%04d_" % (macro_displacement)
           name_nomicro = self.prefdir + 'inp_nomicro%s' % label_nomicro

           frag_1_nomicro.make_input_with(frag_2_nomicro, name=name_nomicro, label='Displacement %s' % label_nomicro, 
                                          xyz=True, gau=True)

           for micro_displacement in range(1,self.n_micro+1):

               # --- [2] relative orientation
               rel = frag_1_nomicro.relative_orientation_between(frag_2_nomicro)
               data_inter = "%6d" % I
               data_inter+= 3*"%14.6E" % rel 
               data_inter+= "\n"
               #
               data_file_interfragment.write(data_inter)

               # --- displacements
               dq = frag_1.generate_random_dq(scale=self.s_micro)
               frag_1.microdisplacement(+dq)
               frag_2.microdisplacement(-dq)

               label = "_%04d_%04d_" % (macro_displacement, micro_displacement)
               name = self.prefdir + 'inp_micro%s' % label

               print(" * Making %s displacement" % label)

               frag_1.make_input_with(frag_2, name=name, label='Displacement %s' % label, xyz=True)

               # --- data files
               # --- [1] micro-displacements
               nq1 = len(frag_1.micro_displacement)
               nq2 = len(frag_2.micro_displacement)

               data_micro = "%6d" % I
               data_micro+= nq1*"%14.6E" % tuple(frag_1.micro_displacement)
               data_micro+= nq2*"%14.6E" % tuple(frag_2.micro_displacement)
               data_micro+= "\n"
               #
               data_file_microdisplacements.write(data_micro)
   
              # # --- [3] electrostatic potential
              # pot_1 = frag_1.electrostatic_potential_from(frag_2)
              # pot_2 = frag_2.electrostatic_potential_from(frag_1)
              # data_elect = "%6d" % I
              # data_elect+= frag_1.natoms*"%14.6E" % tuple(pot_1) 
              # data_elect+= frag_2.natoms*"%14.6E" % tuple(pot_2) 
              # data_elect+= "\n"
              # #
              # data_file_electrostatic.write(data_elect)

              # pot_1 = frag_1_nomicro.electrostatic_potential_from(frag_2_nomicro)
              # pot_2 = frag_2_nomicro.electrostatic_potential_from(frag_1_nomicro)
              # data_elect = "%6d" % I
              # data_elect+= frag_1_nomicro.natoms*"%14.6E" % tuple(pot_1) 
              # data_elect+= frag_2_nomicro.natoms*"%14.6E" % tuple(pot_2) 
              # data_elect+= "\n"
              # #
              # data_file_electrostatic_0.write(data_elect)

              # # --- [4] energy data via psi4
              # en = self._calculate_all_symmetric_case(frag_1.make_input_with(frag_2, 
              #                                         psi=True, no_file=True))
              # data_energy = "%6d" % I
              # data_energy+= "%14.6E" % en.v_0
              # data_energy+= "%14.6E" % en.e_ex
              # data_energy+= "%14.6E" % en.v_eet
              # data_energy+= "\n"

              # data_file_energy.write(data_energy)

              # #
               I += 1


       #
       data_file_microdisplacements.close()
       data_file_interfragment     .close()
       data_file_electrostatic     .close()
       data_file_electrostatic_0   .close()
       data_file_energy            .close()

   def _calculate_fed(self, log): #OK
       c = 1./libbbg.units.UNITS.HartreeToElectronVolt
       E_0 = 0.0 # gefp.core.utilities.read_energy_from_gaussian_log(log)
       E_1 = E_0 + gefp.core.utilities.read_transition_energy_from_gaussian_log(log, 1, False) * c
       E_2 = E_0 + gefp.core.utilities.read_transition_energy_from_gaussian_log(log, 2, False) * c

       fed = gefp.core.driver.FED(log, basis_label='BASIS', max_states=2)
       v_00 = E_0
       v_12 = fed.coupling(1,2, symmetrize=True)
       v_11, v_22 = self._compute_diabatic_pes(E_1, E_2, v_12)
       return E_0, E_1, E_2, v_00, v_11, v_22, v_12

   def _calculate_all_symmetric_case(self, psi, state_1=STATE_1, state_2=STATE_2, state_mon=STATE_MON): #TODO
       mol = psi4.geometry(psi)
       mol1= mol.extract_subsets(1,2)
       mol2= mol.extract_subsets(2,1)

       # SCF: ground state energy
       v0, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)
       v01, wfn1 = psi4.energy('scf', molecule=mol1, return_wfn=True)
      #v02, wfn2 = psi4.energy('scf', molecule=mol2, return_wfn=True)
       dv_0 = v0 - v01*2 #- v02

       # CIS: diabatic site energy and coupling
       cis = oepdev.CISComputer.build("RESTRICTED", wfn, psi4.core.get_options(), "RHF")
       cis.compute()
       cis.clear_dpd()
       psi4.core.clean()
       E = cis.eigenvalues().to_array(dense=True)

       dv_eet = 0.5*(E[state_2-1] - E[state_1-1])
       e_ex   = 0.5*(E[state_2-1] + E[state_1-1])

      # cis1 = oepdev.CISComputer.build("RESTRICTED", wfn1, psi4.core.get_options(), "RHF")
      # cis1.compute()
      # cis1.clear_dpd()
      # psi4.core.clean()
      # E1 = cis1.eigenvectors().to_array(dense=True)

      # e_ex1 = E1[state_mon-1]
       e_ex1 = self.data_0.e_ex

       de_ex = e_ex - e_ex1
      # cis2 = oepdev.CISComputer.build("RESTRICTED", wfn2, psi4.core.get_options(), "RHF")
      # cis2.compute()
      # cis2.clear_dpd()
      # psi4.core.clean()
      # E2 = cis2.eigenvectors().to_array(dense=True)

       return DataSymmetric(dv_0, de_ex, dv_eet)


   def make_densities(self): #OK
       print("Parsing Gaussian input files")
       inp_files = glob.glob("%s/inp_micro*.inp" % self.directory); inp_files.sort()
       inp0_files = glob.glob("%s/inp_nomicro*.inp" % self.directory); inp0_files.sort()

       data_file_potential_diabatic           = open('data_diaba.dat','w')
       data_file_potential_adiabatic          = open('data_adiab.dat','w')
       data_file_potential_0_diabatic         = open('data_diaba_0.dat', 'w')
       data_file_potential_0_adiabatic        = open('data_adiab_0.dat', 'w')

       I = 1
       for inp in inp_files: 

           log = inp[:-3] + 'log'

           print(" File: %s" % log)

           pred, stem, macro_displ, micro_displ, ext = log.split('_')
           log_macro = '_'.join([pred, 'nomicro', macro_displ, ext])

           try:
          #if 1:
              # read data                                                                                 
              E_0, E_1, E_2, V_00, V_11, V_22, V_12 = self._calculate_fed(log)
              e_0, e_1, e_2, v_00, v_11, v_22, v_12 = self._calculate_fed(log_macro)

              # diabatic data
              data_1 = "%6d" % I
              data_1+= 4*"%14.6E" % (V_00, V_11, V_22, V_12)
              data_1+= "\n"
              #
              data_file_potential_diabatic.write(data_1)
                                                                                                          
              # adiabatic data
              data_2 = "%6d" % I
              data_2+= 3*"%14.6E" % (E_0, E_1, E_2)
              data_2+= "\n"
              #
              data_file_potential_adiabatic.write(data_2)

              # diabatic data (0)
              data_1 = "%6d" % I
              data_1+= 4*"%14.6E" % (v_00, v_11, v_22, v_12)
              data_1+= "\n"
              #
              data_file_potential_0_diabatic.write(data_1)
                                                                                                          
              # adiabatic data (0)
              data_2 = "%6d" % I
              data_2+= 3*"%14.6E" % (e_0, e_1, e_2)
              data_2+= "\n"
              #
              data_file_potential_0_adiabatic.write(data_2)


           except:
              print(" Log file No. %d named %s not found or no data found" % (I, log))

           #
           I += 1

       #
       data_file_potential_diabatic.close()
       data_file_potential_adiabatic.close()
       data_file_potential_0_diabatic.close()
       data_file_potential_0_adiabatic.close()
       return


#TODO: Add nicer interface
if __name__ == '__main__':

   freq = 'freq/ccsd/e_ccsd_freq.log'
   sampl = Samples(freq,       atid=[2,1,6],
                               n_macro=120, n_micro=1,
                               mindist=3.5, mintransl=3.0, transl_span=9.0,
                               s_micro=0.0, rot_amplitude=0.0,
                               selected_modes=[2,])
   
   sampl.make_inputs()
