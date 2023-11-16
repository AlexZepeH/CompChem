#!/usr/bin/python3
import gefp, psi4, libbbg, sys, oepdev, numpy, scipy.linalg, math, os, scipy, string, re
from gefp.math.differentials import FiniteDifferenceTerm
from abc import ABC, abstractmethod

# input templates
class InputTemplate(string.Template): delimiter = '@'
template_gaussian = InputTemplate(open('/home/alejandro/templates/template_gaussian.dat').read())
template_sbatch   = InputTemplate(open('templates/template_sbatch.dat').read())
numpy.random.seed(400)

numpy.set_printoptions(precision=8, threshold=None, edgeitems=None, 
                       linewidth=None, suppress=True, nanstr=None, 
                       infstr=None, formatter=None, sign='+', 
                       floatmode='fixed', legacy=None)

# settings for Ethylene (LVC) - obsolete. Ignore it inless need LVC - then need to improve implementation of this (current is only with Psi4 on-the-fly, no Gaussian log file readout)
#SINGLET_STATES = [ 2, 4, 8, 9, 10, 13, 18, 19, 21, 22, 23, 26, 29] # 6-31G*
SINGLET_STATES = [ 3, 5, ] #6, 8, 11, 13, 15, 16, 21, 25, 26, 27, 28, 30] # aug-cc-pVDZ
N_SINGLET_STATES = len(SINGLET_STATES) + 1

SINGLET_STATES = numpy.array(SINGLET_STATES, dtype=int) - 1


def set_template_gaussian(f): 
    global template_gaussian
    template_gaussian = InputTemplate(open(f).read())
def set_template_sbatch  (f): 
    global template_sbatch
    template_sbatch   = InputTemplate(open(f).read())

class Fragment:
   def __init__(self, xyz, lvec, atoms, charges, idx=[2,1,6], selected_modes=None):
       self.name= "undefined"
       self.shortname = "none"
       self.charge = 0
       self.multiplicity = 1

       self.xyz = xyz.copy()
       self.lvec= lvec.copy()
       self.atoms = numpy.array(atoms, 'str')
       self.charges = charges.copy()
       self.natoms = len(xyz)
       self.microdisplaced = False
       self.nmodes_all = 3*self.natoms - 6
       if selected_modes is None:
          self.nmodes_act = self.nmodes_all
          self.selected_modes = numpy.arange(self.nmodes_all)
       else:
          self.selected_modes = numpy.array(selected_modes, int) - 1
          self.nmodes_act = len(self.selected_modes)
       if self.nmodes_act > 0: assert self.nmodes_act == len(self.lvec)
       self.micro_displacement = None
       self.orient_idx = numpy.array(idx) - 1
   @classmethod
   def empty(cls):
       xyz = numpy.zeros((0,3), numpy.float64)
       lvec= numpy.zeros((0,0,3), numpy.float64)
       atoms= numpy.array([], str)
       charges = numpy.zeros(0, numpy.float64)
       return Fragment(xyz, lvec, atoms, charges, idx=[1,2,3], selected_modes=None)
   def clone(self):
       return Fragment(self.xyz, self.lvec, self.atoms, self.charges, self.orient_idx+1, self.selected_modes+1)
   def cog(self):
       return self.xyz.sum(axis=0) / self.natoms
   def move_to_origin(self):
       self.translate(-self.cog())
   def translate(self, t):
       self.xyz = self.xyz + t
   def rotate(self, r):
       self.xyz = self.xyz @ r 
       self.lvec= self.lvec @ r
   def make_translated(self, t):
       return Fragment(self.xyz+t, self.lvec, self.atoms, self.charges, self.orient_idx+1, self.selected_modes+1)
   def make_rotated(self, r):
       xyz = self.xyz @ r 
       lvec= self.lvec @ r
       return Fragment(xyz, lvec, self.atoms, self.charges, self.orient_idx+1, self.selected_modes+1)
   def superimpose(self, xyz, suplist=None):
       if suplist is not None:
          xyz_i = self.xyz[suplist]
          xyz_f =      xyz[suplist]
       else:
          xyz_i = self.xyz
          xyz_f =      xyz
       sup = gefp.math.matrix.Superimposer()           
       sup.set(xyz_f,xyz_i)
       sup.run()
       rms = sup.get_rms()
       rot, transl = sup.get_rotran()
       self.xyz = self.xyz @ rot + transl
       self.lvec= self.lvec @ rot

   def make_superimposed(self, xyz, suplist=None):
       if suplist is not None:
          xyz_i = self.xyz[suplist]
          xyz_f =      xyz[suplist]
       else:
          xyz_i = self.xyz
          xyz_f =      xyz
       sup = gefp.math.matrix.Superimposer()           
       sup.set(xyz_f,xyz_i)
       sup.run()
       rms = sup.get_rms()
       rot, transl = sup.get_rotran()
       xyz_new = self.xyz @ rot + transl
       lvec_new= self.lvec @ rot
       return Fragment(xyz_new, lvec_new, self.atoms, self.charges, self.orient_idx+1, self.selected_modes+1)

   def microdisplacement(self, dq=None, scale=1.0):
       if dq is None:
          dq = scale*2.0*(numpy.random.random(self.nmodes_act) - 0.5)
       dx = numpy.einsum("i,iab->ab", dq, self.lvec)
       self.xyz += dx
       self.microdisplaced = True
       self.micro_displacement = dq.copy()
   def make_microdisplaced(self, dq=None, scale=1.0):
       if dq is None:
          dq = scale*2.0*(numpy.random.random(self.nmodes_act) - 0.5)
       dx = numpy.einsum("i,iab->ab", dq, self.lvec)
       xyz = self.xyz + dx
       frag = Fragment(xyz, self.lvec, self.atoms, self.charges, self.orient_idx+1, self.selected_modes+1)
       frag.microdisplaced = True
       frag.micro_displacement = dq.copy()
       return frag

   @classmethod
   def rotation_matrix_around_axis(cls, u, theta):
       "Rotation matrix around axis u by theta in radians"
       U = u/numpy.linalg.norm(u)
       x,y,z = U
       s = numpy.sin(theta)
       c = numpy.cos(theta)
       C = 1.0 - c
       R = numpy.array([[x*x*C+  c, x*y*C-z*s, x*z*C+y*s],
                        [y*x*C+z*s, y*y*C+  c, y*z*C-x*s],
                        [z*x*C-y*s, z*y*C+x*s, z*z*C+  c]])
       return R


   def torsion_displacement(self, atoms, axis, theta=0.0, scale=1.0):
       "Rotate atoms around the axis by theta degrees. Atoms given in Python convention"
       if theta is None:
          theta = scale*2.0*(numpy.random.random() - 0.5)

       #def ROT_AXIS(u, theta):
       #    "Rotation matrix around axis u by theta in radians"
       #    U = u/numpy.linalg.norm(u)
       #    x,y,z = U
       #    s = numpy.sin(theta)
       #    c = numpy.cos(theta)
       #    C = 1.0 - c
       #    R = numpy.array([[x*x*C+  c, x*y*C-z*s, x*z*C+y*s],
       #                     [y*x*C+z*s, y*y*C+  c, y*z*C-x*s],
       #                     [z*x*C-y*s, z*y*C+x*s, z*z*C+  c]])
       #    return R

       zero = self.xyz[axis[0]]
       self.translate(-zero)

       xyz = self.xyz.copy()
      #xyz-= zero

       u = xyz[axis[1]] - xyz[axis[0]]
       p = xyz[atoms]

       p = p @ self.rotation_matrix_around_axis(u, theta*math.pi/180.0)
      #p+= zero

       for i,a in enumerate(atoms):
           self.xyz[a] = p[i]
       self.translate(zero)
       pass

   def coord_entry(self, *others):
       log = ""
       for i in range(self.natoms):
           log += "%4s" % self.atoms[i]
           log += "%14.6f %14.6f %14.6f\n" % tuple(self.xyz[i]) 
      #if others is not None:
       if 1:
          for other in others:
              log += other.coord_entry()
       return log
   def __repr__(self):
       log = self.coord_entry()
       return str(log)
   def distance_matrix(self, other):
       D = numpy.zeros((self.natoms, other.natoms))
       for i in range(self.natoms): 
           ri = self.xyz[i]
           for j in range(other.natoms):
               rj = other.xyz[j]
         
               rij = math.sqrt( ((ri - rj)*(ri - rj)).sum() )

               D[i,j] = rij
       return D
   def minimum_distance(self, other):
       D = self.distance_matrix(other)
       return D.min()
   def make_input_with(self, *others, name='test', label='test label', 
                             xyz=False, psi=False, gau=False, no_file=False, append=False):
       if gau:
          log = template_gaussian.substitute(DATA=self.coord_entry(*others), LABEL=label, 
                                             CHK=os.path.basename(name+'.chk'))                 
          inp = open(name+'.inp','w')
          inp.write(log)
          inp.close()
          #
          log = template_sbatch.substitute(CHK=os.path.basename(name+'.chk'), INP=os.path.basename(name+'.inp'), 
                                           JOB=os.path.basename(name))
          inp = open(name+'.sh','w')
          inp.write(log)
          inp.close()
       if xyz:
          if append: xyz = open(name+'.xyz','a')
          else:      xyz = open(name+'.xyz','w')
          nat = self.natoms + sum([other.natoms for other in others])
          xyz.write("%d\n\n" % nat)
          xyz.write(self.coord_entry(*others))
          xyz.close()
       if psi:
          psis=[]
          for mol in [self,]+list(others):
              log  = "%d %d\n" % (mol.charge, mol.multiplicity)
              log += mol.coord_entry()
              log += "symmetry c1\nunits angstrom\nno_reorient\nno_com\n"
              psis.append(log)
          log = '\n'+'--\n'.join(psis)
          if no_file: return log
          psi = open(name+'.psi','w')
          psi.write(log)
          psi.close()

   @classmethod
   def generate_random_rotation_around_axis(cls, u, rot_amplitude, all=False):
       "Amplitude given by degrees"
       theta = rot_amplitude*2.0*(numpy.random.random() - 0.5)  * math.pi/180.0
       r = cls.rotation_matrix_around_axis(u, theta)
       if all: return r, theta
       return r

   @classmethod
   def generate_random_translation(cls, transl_span, min_transl, u0=None):
       if u0 is None:
          u = numpy.random.random(3) - 0.5 
          u/= numpy.linalg.norm(u)
       else:
          u = u0.copy()
       t = u * (transl_span*numpy.random.random() + min_transl)
       return t
         
   def generate_random_rottransl(self, transl_span, rot_amplitude, min_transl, atid=False): #OK
       # rotation
       r = self.random_rotation_matrix(amplitude=rot_amplitude)
       # translation
       if atid: #is not None:
          atid = self.orient_idx
          v1 = self.xyz[atid[1]] - self.xyz[atid[0]]
          v2 = self.xyz[atid[2]] - self.xyz[atid[0]]
          u0 = numpy.cross(v1, v2); u0/= numpy.linalg.norm(u0)
       else:
          u0 = numpy.random.random(3); u0/= numpy.linalg.norm(u0)
       t = u0 * (transl_span*numpy.random.random() + min_transl)
       return r, t
   def generate_random_dq(self, scale=1.0):
       dq = scale*2.0*(numpy.random.random(self.nmodes_act) - 0.5)
       return dq
   def make_macrodisplaced(self, transl_span=6.0, atid=False, rot_amplitude=30.0, norot=False, min_transl=3.0,
                                 rot_trans=None): #OK
       # determine rotation and translation data
       if rot_trans is None:
          r, t = self.generate_random_rottransl(transl_span, rot_amplitude, min_transl, atid)
          if norot: r = numpy.identity(3)
       #
       else:
          r, t = rot_trans
       # second apply rotation
       frag = self.clone()
       cog = frag.cog()
       frag.move_to_origin()
       frag.rotate(r)
       frag.translate(cog)
       # first apply translation
       frag.translate(t)
       return frag
   def random_rotation_matrix(self, amplitude):
       angles = amplitude*2.*(numpy.random.random(3) - 0.5)
       R = scipy.spatial.transform.Rotation.from_euler('zxy', angles, degrees=True)
       rot= R.as_dcm()
       return rot
   def random_rotation_opposite_matrices(self, amplitude, random_axis=False):
       if not random_axis:
          raise NotImplementedError("This is not implemented yet")
          # the below code is wrong: TODO (fix or remove)
          angles_1 = amplitude*2.*(numpy.random.random(3) - 0.5)
          angles_1[1] = angles_1[2] = 0.0
          angles_2 = angles_1.copy()
         #angles_2*= -1.0 # --> does NOT generate symmetric dimers
          angles_2[0] = -angles_1[0] # --> does NOT generates symmetric dimers
         #angles_2[2] = -angles_1[2] # --> does NOT generates symmetric dimers
                                                                                           
          R1 = scipy.spatial.transform.Rotation.from_euler('xyz', angles_1, degrees=True)
          rot1= R1.as_dcm()
          R2 = scipy.spatial.transform.Rotation.from_euler('xyz', angles_2, degrees=True)
          rot2= R2.as_dcm().T
       else:
          def ROT(u, t):
              "Rotation matrix around axis u (unit vector) by angle t (radians)"
              u /= numpy.linalg.norm(u)
              ct = math.cos(t); st = math.sin(t); uu = numpy.outer(u,u)
              ux = numpy.zeros((3,3));
              ux[0,1] = -u[2]
              ux[1,0] = +u[2]
              ux[0,2] = +u[1]
              ux[2,0] = -u[1]
              ux[1,2] = -u[0]
              ux[2,1] = +u[0]
              R = uu*(1.0 - ct) + ct * numpy.identity(3) + st * ux
              return R
          t = amplitude * numpy.random.random() * math.pi / 180.0
          cog, axis, plane = self.orientation()
          n = plane

          u1= numpy.random.random(3) - 0.5; u1/= numpy.linalg.norm(u1)
          u2= 2.0 * (n @ u1) * n - u1

          rot1 = ROT(u1, +t)
          rot2 = ROT(u2, +t)
       return rot1, rot2
   def orientation(self, model='oriented_plane'):
       "Derive principal position-orientation information"
       if model=='oriented_plane':
          v     = self.xyz[self.orient_idx]
          t     = v[1] - v[0]; t/= numpy.linalg.norm(t)
          p     = v[2] - v[0]; p/= numpy.linalg.norm(p)
          q     = numpy.cross(t, p); q/= numpy.linalg.norm(q)

          cog   = self.cog()
          axis  = t
          plane = q
          return cog, axis, plane
       else: raise NotImplementedError

   def relative_orientation_between(self, other, units='degrees'):
       cog1, axis1, plane1 = self.orientation()
       cog2, axis2, plane2 =other.orientation()
       # distance between cog's
       r = math.sqrt(((cog1 - cog2)*(cog1 - cog2)).sum())
       # angle between capital axes
       A   = axis1 @ axis2
       B   = plane1@ plane2
       if A > 1.0: A= 1.0
       if A <-1.0: A=-1.0
       if B > 1.0: B= 1.0
       if B <-1.0: B=-1.0
       a = abs(min(math.acos(A), math.acos(-A)))
       # angle between capital planes
       b = abs(min(math.acos(B), math.acos(-B)))
       if units == 'degrees':
          a *= 180.0/math.pi
          b *= 180.0/math.pi
       rel = (r, a, b)
       return rel

   def electrostatic_potential_from(self, *others):
       "Electrostatic potential values on atoms of self due to other's charges"
       p = numpy.zeros(self.natoms, dtype=numpy.float64)
       for i in range(self.natoms):
           ri = self.xyz[i]
           v = 0.0
           for other in others:
             for j in range(other.natoms):
               rj = other.xyz[j]

               rij = math.sqrt( ((ri - rj)*(ri - rj)).sum() )
               v += other.charges[j] / rij
           p[i] = v
       return p



class CISData:
  "This is obsolete - please do not use unless know what you are doing."
  def __init__(self, cis, ground_state_energy, Da):
      E = cis.eigenvalues().to_array(dense=True) + ground_state_energy
     #U = cis.eigenvectors().to_array(dense=True)
     #f = [ cis.oscillator_strength(x) for x in range(cis.nstates()) ]
     #T = [ cis.Ta_ao(x).to_array(dense=True)*math.sqrt(2.0) for x in range(cis.nstates()) ] 
     #T = [ cis.Ta_ao(x).to_array(dense=True) + cis.Tb_ao(x).to_array(dense=True) for x in range(cis.nstates()) ] 

      T = [ cis.Ta_ao(x).to_array(dense=True)*math.sqrt(2.0) for x in SINGLET_STATES ] 
      E = E[SINGLET_STATES]

      # append the ground state
      T = [Da, ] + T
      E = numpy.concatenate(([ground_state_energy,], E ))

      # save
      self.V_adiabatic = numpy.diag(E)
      self.ground_state_energy = ground_state_energy
      self.cis = cis
      self.bfs = cis.get_basis()
      self.T = T
      self.Da = Da.copy()

class TDData(libbbg.units.UNITS):
  ""
  def __init__(self, log, basis, nstates=3):
      self.n          = nstates   # number of excited states
      self.density    = {}        # AO density matrices
      self.energy     = None      # excitation energies
      self.mol        = None      # Psi4 Molecule object
      self.bfs        = None      # Psi4 BasisSet object
      self.log        = log
      self.basis      = basis
      self.is_ref     = None      # is it a reference?
      self._init()

  def _init(self):
      convert    = gefp.math.matrix.convert_density_matrix_from_gaussian_to_psi4
      read_gener = gefp.core.utilities.read_energy_from_gaussian_log
      read_ener  = gefp.core.utilities.read_transition_energy_from_gaussian_log
      read_dens  = gefp.core.utilities.read_dmatrix_from_gaussian_log
      read_trans = gefp.core.utilities.read_transition_dmatrix_from_gaussian_log

      if isinstance(self.log, list):
         self.is_ref = True

         self.mol = gefp.core.utilities.psi_molecule_from_gaussian_log(self.log[0])
         self.bfs = psi4.core.BasisSet.build(self.mol, 'G09-BASIS', self.basis, puream=False, quiet=True)  

         self.energy = numpy.zeros(self.n)

         nh = int(self.n/2)
         for i in range(nh):
             self.density[(0,i+1)] = convert( read_trans(self.log[0], i+1), self.bfs ) 
             self.energy[i]        = read_ener(self.log[0], i+1, False) #/ self.HartreeToElectronVolt

             self.density[(0,i+1+nh)] = convert( read_trans(self.log[1], i+1), self.bfs )
             self.energy[i+nh]        = read_ener(self.log[1], i+1, False) #/ self.HartreeToElectronVolt

      else:
         self.is_ref = False

         self.mol = gefp.core.utilities.psi_molecule_from_gaussian_log(self.log)
         self.bfs = psi4.core.BasisSet.build(self.mol, 'G09-BASIS', self.basis, puream=False, quiet=True)  
                                                                                                          
         self.energy = numpy.zeros(self.n)
         for i in range(self.n):
             self.density[(0,i+1)] = convert( read_trans(self.log, i+1), self.bfs )  
             self.energy[i]        = read_ener(self.log, i+1, False) #/ self.HartreeToElectronVolt
        
      return 


class TDDiabatizer:
  """Diabatization in DCBS for monomers. 

 Notes:
  * Purely translational transformations between monomers to form desired system are allowed (no explicit superimposition).
"""
  def __init__(self, reference_state):
      self.state_ref = reference_state
      self.n         = self.state_ref.n

      self.mints = psi4.core.MintsHelper(self.state_ref.bfs)

  def overlap(self, state1, state2):#OK
      "Calculates the overlap matrix between the excited states with respect to the reference"

      # substitute the AO overlap if necessary
      if state1.is_ref:
         s_12 = self.mints.ao_overlap(state2.bfs, state2.bfs).to_array(dense=True)
      elif state2.is_ref:
         s_12 = self.mints.ao_overlap(state1.bfs, state1.bfs).to_array(dense=True)
      else:
         s_12 = self.mints.ao_overlap(state1.bfs, state2.bfs).to_array(dense=True)

     #print(s_12)

     #s_12 = numpy.identity(state1.bfs.nbf())

      S = numpy.zeros((state1.n,state2.n))
      for i in range(state1.n):
          T_0i = state1.density[(0,i+1)]
          for j in range(state2.n):
              T_0j = state2.density[(0,j+1)]
              S[i,j] = (T_0i @ s_12 @ T_0j.T @ s_12.T).trace()

      return S

  def diabatize(self, state):#OK
      S = self.overlap(self.state_ref, state)
     #print(S)
      SSm= scipy.linalg.fractional_matrix_power(S@S.T, -0.50)
      U = S.T @ SSm
     #print(U.shape)
      V = U.T @ numpy.diag(state.energy) @ U
     #print("U=", U)
      return V

def make_dimer_from_monomers(dimer, monomer1, monomer2=None, suplist1=None, suplist2=None, 
                                    start=0, offset=0, return_rotran=False, verbose=False): #OK
    """Make a dimer molecule out of two monomers by superimposing them on a target dimer. 
 Target dimer does not need to have defined fragments but molecules need to be placed
 one after another, possibly with other atoms in between or in the begnning of dimer definition.

 Notes:
  * Superimposition lists are given in normal numbering convention (starting from 1)
  * If there is n atoms that are not part of monomer1, set start=n.
  * Set offset if there are unwanted atoms between monomer1 and monomer2 
    in the target dimer object. Offset=0 means the target dimer is composed
    of the monomers only.
"""
    BohrToAngstrom = libbbg.units.UNITS().BohrToAngstrom

    # determine the monomers to superimpose
    if monomer2 is None:
       monomer2 = monomer1
       suplist2 = suplist1

    # initialize the header and footer of Psi4 molecule entry
    header = "%d %d\n" % (monomer1.molecular_charge()+monomer2.molecular_charge(), monomer1.multiplicity()+monomer2.multiplicity()-1)
    footer = "units angstrom\nsymmetry c1\nno_reorient\nno_com\n"

    # prepare the suplists (in Python convention)
    if suplist1 is None: 
      _suplist1_ = numpy.arange(monomer1.natom())
    else:
      _suplist1_ = numpy.array(suplist1, int) - 1
    if suplist2 is None: 
      _suplist2_ = numpy.arange(monomer2.natom())
    else:
      _suplist2_ = numpy.array(suplist2, int) - 1

    # read the coordinates of target dimer and each of monomers to superimpose
    xyz_dimer = dimer.geometry().to_array(dense=True)[start:]
    xyz_monomer1_0 = monomer1.geometry().to_array(dense=True)
    xyz_monomer2_0 = monomer2.geometry().to_array(dense=True)

    # determine the parts of a dimer corresponding to the monomers
    xyz_monomer1 = xyz_dimer[:monomer1.natom()].copy()
    xyz_monomer2 = xyz_dimer[monomer1.natom() + offset:].copy()

    # superimpose the monomers
    superimposer = gefp.math.matrix.Superimposer()

    superimposer.set(xyz_monomer1[_suplist1_], xyz_monomer1_0[_suplist1_])
    superimposer.run()
    rms1 = superimposer.get_rms()
    r1, t1 = superimposer.get_rotran()

    superimposer.set(xyz_monomer2[_suplist2_], xyz_monomer2_0[_suplist2_])
    superimposer.run()
    rms2 = superimposer.get_rms()
    r2, t2 = superimposer.get_rotran()
    if verbose:
       print(" RMS of superimposition: %14.6f %14.6f [a.u.]" % (rms1, rms2))

    # transform the coordinates of the monomers
    xyz_monomer1_sup = xyz_monomer1_0 @ r1 + t1
    xyz_monomer2_sup = xyz_monomer2_0 @ r2 + t2

    # prepare the coordinate entries for monomers
    coord_entries = ["", ""]
    coord_xyzs    = [xyz_monomer1_sup, xyz_monomer2_sup]
    for i, monomer in enumerate([monomer1, monomer2]):
        for a in range(monomer.natom()):
            coord_entries[i] += "%4s" % monomer.symbol(a)
            coord_entries[i] += "%14.6f %14.6f %14.6f\n" % tuple(coord_xyzs[i][a] * BohrToAngstrom)

    # put-up the dimer string
    psi_string = header + coord_entries[0] + coord_entries[1] + footer
    dimer_new = psi4.geometry('\n'+psi_string)
    dimer_new.update_geometry()

    if verbose:
       print(" Superimposed dimer:")
       print(dimer_new.save_string_xyz())
       print()

    if return_rotran: return dimer_new, (r1, t1, rms1), (r2, t2, rms2)
    return dimer_new


class State:
  """
 Represents an electronically excited TDDFT/CIS state.

 It is fully characterized by the set of ground-to-excited one-particle transition density matrices.
""" 
  def __init__(self, log=None, basis=None, nstates=1, data=None, standardize_phases=False): #OK
      # Preliminary set-up of local variable (attribute) name space
      self.n          = nstates   # Number of excited states within a given state
      self.density    = {}        # AO density matrices
      self.energy     = None      # Adiabatic Excitation energies relative to the ground state
      self.mol        = None      # Psi4 Molecule object
      self.bfs        = None      # Psi4 BasisSet object
      self.basis      = basis     # Symbol of the basis set (e.g. 6-31G*)

      # Initialize the object
      if   log  is not None: self._init_from_Gaussian(log)
      elif data is not None: self._init_from_dictionary(data)

      # Standardize the sign of the transition densities
      if standardize_phases: self._standardize_phases()

  def _init_from_dictionary(self, data): #OK
      "Initialize the state directly from the data given in Python dictionary with entries: nstates, density, energy, mol, bfs, basis."
      self.n       = data["nstates"]
      self.density = data["density"]
      self.energy  = data["energy"]
      self.mol     = data["mol"]
      self.bfs     = data["bfs"]
      self.basis   = data["basis"]
      pass

  def _init_from_Gaussian(self, log): #OK
      "Initialize the state by reading the Gaussian 16 log file with printed transition density information"
      convert    = gefp.math.matrix.convert_density_matrix_from_gaussian_to_psi4
      read_gener = gefp.core.utilities.read_energy_from_gaussian_log
      read_ener  = gefp.core.utilities.read_transition_energy_from_gaussian_log
      read_dens  = gefp.core.utilities.read_dmatrix_from_gaussian_log
      read_trans = gefp.core.utilities.read_transition_dmatrix_from_gaussian_log

      self.mol = gefp.core.utilities.psi_molecule_from_gaussian_log(log)
      self.bfs = psi4.core.BasisSet.build(self.mol, 'G09-BASIS', self.basis, puream=False, quiet=True)  
                                                                                                       
      self.energy = numpy.zeros(self.n)
      for i in range(self.n):
          self.density[(0,i+1)] = convert( read_trans(log, i+1), self.bfs )  
          self.energy[i]        = read_ener(log, i+1, False) #/ self.HartreeToElectronVolt
      pass 

 #def __matrix_sign_function(self, a):
 #    return a @ scipy.linalg.fractional_matrix_power(a@a, -0.5)

  def __matrix_sign_function(self, a):
      u,s,vh=numpy.linalg.svd(a, full_matrices=False)
      cov = u[:,0] @ vh[0]
      return numpy.sign(cov)
 

  def _standardize_phases(self):
      "Assumes the first basis function is the core s-type function for reference sign to be positive"
      for i in range(self.n):
          T = self.density[(0,i+1)]
          self.density[(0,i+1)] = T * self.__matrix_sign_function(T)
         #print("before= ", T[idx,idx], numpy.sign(T[idx,idx]))
         #self.density[(0,i+1)] *= numpy.linalg.eig(self.__matrix_sign_function(T)[0])[0]
         #T = self.density[(0,i+1)]
         #print("after= ", T[idx,idx], numpy.sign(T[idx,idx]))
      pass

class TDDiabatizerMCBS_General:
  """Diabatization in MCBS for monomers.

 Notes:
  * Superimposition implemented.
  * Internal coordinates of monomers within a target systems can differ from those in reference state monomers
"""
  def __init__(self, monomer_state1, monomer_state2=None): #OK
      self.state_monomer1= monomer_state1
      self.state_monomer2= monomer_state2
      if monomer_state2 is None: self.state_monomer2  = self.state_monomer1
      self.mints         = None

  def overlap(self, state1, state2): #OK
      "Calculates the overlap matrix between two excited states (dimer systems)"

      # calculate AO overlap
      s_12 = self.mints.ao_overlap(state1.bfs, state2.bfs).to_array(dense=True)

      # calculate state overlap approximation
      S = numpy.zeros((state1.n,state2.n))
      for i in range(state1.n):
          T_0i = state1.density[(0,i+1)]
          for j in range(state2.n):
              T_0j = state2.density[(0,j+1)] 
              S[i,j] = (T_0i @ s_12 @ T_0j.T @ s_12.T).trace()

     #S = self._standardize_overlap_matrix(S)
      print("Overlap betwreen states:")
      print(S, numpy.linalg.det(S))
      return S

  def _standardize_overlap_matrix(self, S): #TODO
      R = S.copy()
     #for i in range(len(S)):
     #    R[i] = S[numpy.argmax(S[:,i]**2)]
      R = numpy.array([[+0.9,+0.5],[-0.5,+0.9],])
      R = numpy.array([[+0.82897306, +0.48827542],[-0.49986758, +0.81905073]])

      Sc, sim = libbbg.utilities.order(R,S,start=0,lprint=1)
     # Sc = S.copy()
     # u,s,vh = numpy.linalg.svd(S, full_matrices=False)
     # g = numpy.sign(numpy.linalg.det(vh.T@u.T))
     # s = numpy.diag(s); s[-1,-1] *= g
     ##Sc= u.T @ s @ vh.T
     # Sc= vh.T @ s @ u.T
     # #E,U = numpy.linalg.eig(S)
     # #for i in range(len(E)):
     # #    if abs(E[i]) < 0.0:
     # #       E[i] *= -1.0j
     # #Sc = U.T.conjugate() @ numpy.diag(E) @ U
     ##if numpy.linalg.det(S) < 0.0:
     ##   Sc[:,-1] *= -1.0
     # #for i in range(S.shape[1]):
     # #    if S[i,i] < 0.0:
     # #       Sc[:,i] *= -1.0
      return Sc

  def diabatize(self, target_state, return_all=False, suplist1=None, suplist2=None, start=0, offset=0, remove1=0, remove2=0): #OK
      """Calculates the adiabatic-to-diabatic transformation matrix and diabatic potential matrix.

 Definition:

  U = U(Adiabatic,Diabatic)
"""
      # Initialize Mints if not done so yet
      if self.mints is None: self.mints = psi4.core.MintsHelper(target_state.bfs)

      # Construct the reference diabatic state from the monomers based on the target dimer state
      state_ref = self.get_reference_state(target_state, self.state_monomer1, self.state_monomer2, 
                                                         suplist1=suplist1, suplist2=suplist2,
                                                         start=start, offset=offset, remove1=remove1, remove2=remove2)

      # Calculate the state overlap between reference and the target
      S = self.overlap(state_ref, target_state)

      # Perform the diabatization 
      SSm= self.regularized_matrix_power(S@S.T, -0.5)
      U = S.T @ SSm
      V = U.T @ numpy.diag(target_state.energy) @ U

      if return_all: return V, U
      return V

  @classmethod
  def regularized_matrix_power(cls, matrix, power):
      "It removes the small negative value noise in eigenvalues"
      d,u = numpy.linalg.eig(matrix)
      if power<0.0: d = numpy.abs(d)
      dp = d**power
      matrix_power = u @ numpy.diag(dp) @ u.T
      return matrix_power

  @classmethod
  def get_reference_state(cls, state_dimer, state_monomer1, state_monomer2, suplist1=None, suplist2=None, start=0, offset=0, remove1=0, remove2=0): #OK
      """
 Construct the reference (diabatic) state of a dimer from adiabatic dimer state and reference adiabatic monomer states.

 Notes:
  * The reference (diabatic) state is here defined as the Hadamard sum of MCBS (superimposed) transition densities.
    of each monomer. Superimposition, perform separately for each monomer, is based on the atomic coordinates comparison 
    between state_dimer and state_monomer.
"""

      # some sanity checks
      assert state_dimer.basis == state_monomer1.basis == state_monomer2.basis, "Error: Basis sets for dimer and monomers seem not compatible!"

      # create monomer molecule objetcs that are rotated and translated accordingly and combine them in a single dimer object
      mol_reference, R1, R2 = gefp.core.driver.make_dimer_from_monomers(state_dimer.mol, state_monomer1.mol, state_monomer2.mol, 
                                                                        suplist1=suplist1, suplist2=suplist2, remove1=remove1, remove2=remove2, 
                                                                        start=start, offset=offset, verbose=True, return_rotran=True)

      # extract the superimposition data
      r1, t1, rms1 = R1
      r2, t2, rms2 = R2

      # rotate the transition densities
      density_reference = {}; K1 = state_monomer1.bfs.nbf(); K2 = state_monomer2.bfs.nbf(); N = K1 + K2
      for i in range(state_monomer1.n): 
          I_1      = 2*i + 0
          I_2      = 2*i + 1
          T_1      = numpy.zeros((N, N)) 
          T_2      = numpy.zeros((N, N)) 
          T0_mon1  = state_monomer1.density[(0,i+1)].copy()
          T0_mon2  = state_monomer2.density[(0,i+1)].copy()
         #T0_mon1 *= numpy.sign(T0_mon1[0,0])
         #T0_mon2 *= numpy.sign(T0_mon2[0,0])
          T_1[:K1,:K1] = gefp.math.matrix.rotate_ao_matrix(T0_mon1, r1, state_monomer1.bfs, orbitals=False, density=True, return_rot=False)
          T_2[K1:,K1:] = gefp.math.matrix.rotate_ao_matrix(T0_mon2, r2, state_monomer2.bfs, orbitals=False, density=True, return_rot=False)
         
          density_reference[(0,I_1+1)] = T_1
          density_reference[(0,I_2+1)] = T_2

      energy_reference = state_monomer1.energy + state_monomer2.energy #TODO: its wrong order!

      # create BasisSet object for the generated dimer
      bfs_reference = psi4.core.BasisSet.build(mol_reference, 'G09-BASIS', state_monomer1.basis, puream=False, quiet=True)

      # set-up the reference object
      state_reference = State(data={"nstates": state_monomer1.n + state_monomer2.n , 
                                    "density": density_reference  ,
                                    "energy" : energy_reference   ,
                                    "basis"  : state_monomer1.basis,
                                    "mol"    : mol_reference      ,
                                    "bfs"    : bfs_reference      }, standardize_phases=False)

      return state_reference



     
class TDDiabatizerMCBS:
  """Diabatization in MCBS for monomer.

 Notes:
  * Superimposition implemented.
  * Internal coordinates of monomers within a target systems can differ from those in reference state monomers
"""
  def __init__(self, monomer_state): #OK
      self.state_monomer = monomer_state
      self.mints         = None

  def overlap(self, state1, state2): #OK
      "Calculates the overlap matrix between two excited states (dimer systems)"

      # calculate AO overlap
      s_12 = self.mints.ao_overlap(state1.bfs, state2.bfs).to_array(dense=True)

      # calculate state overlap approximation
      S = numpy.zeros((state1.n,state2.n))
      for i in range(state1.n):
          T_0i = state1.density[(0,i+1)]
          for j in range(state2.n):
              T_0j = state2.density[(0,j+1)]
              S[i,j] = (T_0i @ s_12 @ T_0j.T @ s_12.T).trace()

      return S

  def diabatize(self, state, return_all=False, suplist=None): #OK
      """Calculates the adiabatic-to-diabatic transformation matrix and diabatic potential matrix.

 Definition:

  U = U(Adiabatic,Diabatic)
"""
      # Initialize Mints if not done so yet
      if self.mints is None: self.mints = psi4.core.MintsHelper(state.bfs)

      # Construct the reference diabatic state from the monomers based on the target dimer state
      state_ref = self.get_reference_state(state, self.state_monomer, suplist=suplist)

      # Calculate the state overlap between reference and the target
      S = self.overlap(state_ref, state)

      # Perform the diabatization 
      SSm= self.regularized_matrix_power(S@S.T, -0.5)
      U = S.T @ SSm
      V = U.T @ numpy.diag(state.energy) @ U

      if return_all: return V, U
      return V

  @classmethod
  def regularized_matrix_power(cls, matrix, power):
      "It removes the small negative value noise in eigenvalues"
      d,u = numpy.linalg.eig(matrix)
      if power<0.0: d = numpy.abs(d)
      dp = d**power
      matrix_power = u @ numpy.diag(dp) @ u.T
      return matrix_power

  @classmethod
  def get_reference_state(cls, state_dimer, state_monomer, suplist=None): #OK
      """
 Construct the reference (diabatic) state of a dimer from adiabatic dimer state and reference adiabatic monomer states.

 Notes:
  * The reference (diabatic) state is here defined as the Hadamard sum of MCBS (superimposed) transition densities.
    of each monomer. Superimposition, perform separately for each monomer, is based on the atomic coordinates comparison 
    between state_dimer and state_monomer.
"""

      # some sanity checks
      assert state_dimer.bfs.nbf() == state_monomer.bfs.nbf()*2, "Error: Basis sets for dimer and monomers seem not compatible!"

      # create monomer molecule objetcs that are rotated and translated accordingly and combine them in a single dimer object
      mol_reference, R1, R2 = make_dimer_from_monomers(state_dimer.mol, state_monomer.mol, suplist1=suplist, verbose=True, return_rotran=True)

      # extract the superimposition data
      r1, t1, rms1 = R1
      r2, t2, rms2 = R2

      # rotate the transition densities
      density_reference = {}; N = state_dimer.bfs.nbf(); K = state_monomer.bfs.nbf()
      for i in range(state_monomer.n): 
          I_1      = 2*i + 0
          I_2      = 2*i + 1
          T_1      = numpy.zeros((N, N)) 
          T_2      = numpy.zeros((N, N)) 
          T0_mon   = state_monomer.density[(0,i+1)].copy()
          T_1[:K,:K] = gefp.math.matrix.rotate_ao_matrix(T0_mon, r1, state_monomer.bfs, orbitals=False, density=True, return_rot=False)
          T_2[K:,K:] = gefp.math.matrix.rotate_ao_matrix(T0_mon, r2, state_monomer.bfs, orbitals=False, density=True, return_rot=False)
         
          density_reference[(0,I_1+1)] = T_1
          density_reference[(0,I_2+1)] = T_2

      energy_reference = state_monomer.energy * 2 #TODO

      # create BasisSet object for the generated dimer
      bfs_reference = psi4.core.BasisSet.build(mol_reference, 'G09-BASIS', state_monomer.basis, puream=False, quiet=True)

      # set-up the reference object
      state_reference = State(data={"nstates": state_monomer.n*2  , 
                                    "density": density_reference  ,
                                    "energy" : energy_reference   ,
                                    "basis"  : state_monomer.basis,
                                    "mol"    : mol_reference      ,
                                    "bfs"    : bfs_reference      })

      return state_reference


            
class CISPropertyFunctor:
  """
 Functor that implements the CIS property as a function of atomic coordinates.

 Do not use (obsolete), unless you want LVC couplings from on-the-fly EOPDev CIS calculations (not recommended because it is slow and memory consuming).
"""

  def __init__(self, molecule):
      self.molecule = molecule
      self.e, self.wfn = psi4.energy('scf', molecule=molecule, return_wfn=True)

      self.bfs_ref = self.wfn.basisset()
      x = self.molecule.geometry()
      self.XYZ_ref = x.clone()

      self.mints = psi4.core.MintsHelper(self.bfs_ref)
      self._initialized = False

      # perform the CIS calculation for the reference adiabatic/diabatic states
      self.E_ref, self.cis_data_ref = self.__call__(molecule.natom()*3*[0.0,], return_all=True)
      self._initialized = True
     #exit()
     
  
  def reset(self):
      self.molecule.set_geometry(self.XYZ_ref)
      self.molecule.update_geometry()

  def __call__(self, *dx, return_all=False):
      "Do calculation for displaced molecule by dx vector"
      # perturb the atomic structure
      xyz = self.molecule.geometry().to_array(dense=True)
      xyz+= numpy.array(dx).reshape(self.molecule.natom(),3)
      XYZ = psi4.core.Matrix.from_array(xyz)
      self.molecule.set_geometry(XYZ)
      self.molecule.update_geometry()

      # perform adiabatic calculation
      ground_state_energy, wfn = psi4.energy('scf', return_wfn=True, molecule=self.molecule)                         
      Da = wfn.Da().to_array(dense=True) / (2*math.sqrt(2.0))
      cis  = oepdev.CISComputer.build("RESTRICTED", wfn, psi4.core.get_options(), "RHF")
      cis.compute()
      cis.clear_dpd()
      psi4.core.clean()

      cis_data = CISData(cis, ground_state_energy, Da)

      # perform diabatization
      S = self.calculate_overlap(cis_data)
      SSm= scipy.linalg.fractional_matrix_power(S@S.T, -0.50)
      Q = S.T @ SSm
      V = Q.T @ cis_data.V_adiabatic @ Q

      self.reset()
      print("V=",V)
      if not return_all:
        #return V[1,0]
         return V[2,2]
      else:
        #return V[1,0], cis_data
         return V[2,2], cis_data

  def calculate_overlap(self, cis_data):
      "Calculates the overlap matrix between the excited states with respect to the reference"
      n = N_SINGLET_STATES #cis_data.cis.nstates()

      if self._initialized:                                    
         S_01 = self.mints.ao_overlap(self.bfs_ref, cis_data.bfs).to_array(dense=True)
         T_0  = self.cis_data_ref.T
         T_1  =      cis_data    .T
                                                                  
         S = numpy.zeros((n,n))
         for i in range(n):
             for j in range(n):
                 S[i,j] = (T_0[i] @ S_01 @ T_1[j].T @ S_01.T).trace()

      else:
          S = numpy.identity(n)
      print("S=",S)
     #if self._initialized: exit()
      return S

  def zero(self):
      "Returns null element of Function's vector space"
      return 0.0

class FiniteDifferenceScheme_Ethylene(gefp.math.differentials.FiniteDifferenceScheme):
  """
 Finite Difference Central 5-Point Scheme with:
  * 1st and 2nd derivatives of truncation error O4
  * 3rd and 4th derivatives of truncation error 02
"""
  def __init__(self):
      self.implemented_derivatives = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), ]
      super(FiniteDifferenceScheme_Ethylene, self).__init__(self.implemented_derivatives)

  def _add_terms(self):
      "Add terms to the finite difference scheme"
      # --> 1st Derivatives <-- #
      # a1x     
      self._add( (0,), FiniteDifferenceTerm( disps=([-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (0,), FiniteDifferenceTerm( disps=([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a1y                                                                                  
      self._add( (1,), FiniteDifferenceTerm( disps=([ 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (1,), FiniteDifferenceTerm( disps=([ 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a1z                                                                                  
      self._add( (2,), FiniteDifferenceTerm( disps=([ 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (2,), FiniteDifferenceTerm( disps=([ 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a2x                                                                                  
      self._add( (3,), FiniteDifferenceTerm( disps=([ 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (3,), FiniteDifferenceTerm( disps=([ 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a2y                                                                                  
      self._add( (4,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                                                    [ 0, 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (4,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0, 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a2z                                                                                  
      self._add( (5,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0, 0, 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (5,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                                    [ 0, 0, 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )



      # a3x     
      self._add( (6,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (6,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a3y                                                                                  
      self._add( (7,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (7,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a3z                                                                                  
      self._add( (8,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (8,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a4x                                                                                  
      self._add( (9,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add( (9,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a4y                                                                                  
      self._add((10,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, ],  
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add((10,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a4z                                                                                  
      self._add((11,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+2, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add((11,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, ], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+1, 0, 0, 0, 0, 0, 0, ]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )



      # a5x     
      self._add((12,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+2, 0, 0, 0, 0, 0,]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add((12,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+1, 0, 0, 0, 0, 0,]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a5y                                                                                  
      self._add((13,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+2, 0, 0, 0, 0,]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add((13,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+1, 0, 0, 0, 0,]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a5z                                                                                  
      self._add((14,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+2, 0, 0, 0,]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add((14,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+1, 0, 0, 0,]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a6x                                                                                  
      self._add((15,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+2, 0, 0,]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add((15,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+1, 0, 0,]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a6y                                                                                  
      self._add((16,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,],  
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+2, 0,]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add((16,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+1, 0,]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )
      # a6z                                                                                  
      self._add((17,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+2,]), 
                                                     sgns=( 1,-1), coeff=  1.0/12.0 )  )
      self._add((17,), FiniteDifferenceTerm( disps=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,], 
                                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,+1,]), 
                                                     sgns=(-1, 1), coeff=  8.0/12.0 )  )



# optimization
class Optimizer(ABC):
   "Find the equilibrium geometry of the system by gradient search."

   def __init__(self):
       pass

   @classmethod
   def create(cls, *args, model='exact'):
       if model.lower()[0] == 'e':
          return Optimizer_ExactPotential(*args)
       elif model.lower()[0] == 'n':
          return Optimizer_NNPotential(*args)
       else: raise ValueError
       return

   @abstractmethod
   def gradient(self, x):
       pass

   @abstractmethod
   def hessian(self, x):
       pass

   def update(self, x, constrain_z=False):
       G = self.gradient(x)
       if constrain_z:
          G[2] = 0.0
       g  = numpy.sqrt( (G*G).sum() )
       return G, g

   def run(self, x0, conver=1.e-7, constrain_z=False):
       "Gradient Descent Method"
       x1 = x0.copy(); t = 0.001
       
       G1, g1 = self.update(x0, constrain_z)
       dx = -G1 * t; x2 = x1 + dx
       G2, g2 = self.update(x2, constrain_z)
       
       t      = numpy.abs( (x1-x2) @ (G1-G2) ) / numpy.linalg.norm((G1-G2))**2
       
       I = 2
       while g2 > conver:
       
            dx = -G2 * t
       
            x1 = x2.copy() 
            x2 = x2 + dx   
       
            G1 = G2.copy(); g1 = g2
            G2, g2 = self.update(x2, constrain_z)
       
            t  = numpy.abs( (x1-x2) @ (G1-G2) ) / numpy.linalg.norm((G1-G2))**2
            I += 1
            print(" Gradient at I=%3d  is g= %14.8f" % (I+1,g2))
       
       x = x2

       print(" Minimum found after %d iterations." % I)     
       print(" Coordinates: ")
       print(x)
      #print(self.hessian(x))
       return x

   def freq(self, x, mass=None):
       "Frequency analysis"
       if mass is None:
          mass = numpy.ones(x.shape)

       # mass-weighted Hessian matrix
       M = numpy.diag(1./numpy.sqrt(mass))
       H = self.hessian(x)
       H = M @ H @ M

       # vibrational analysis
       w2, L = numpy.linalg.eigh(H)
       w = numpy.sqrt(w2)

       w_cmRec = w * 219474.63
       return w_cmRec, L

class Optimizer_ExactPotential(Optimizer):
   def __init__(self, *args):
       Optimizer.__init__(self)
       #TODO

   def gradient(self, x): #TODO
       G = None
       return G

   def hessian(self, x): pass

class Hamiltonian(libbbg.units.UNITS):
   "Potential Energy Hamiltonian for Quasi-1D Chain"
   def __init__(self, N, 
                      nn_v0, nn_ex, nn_eet,  
                      e_ex=7.608, v_0=0.0, 
                      include_ground_state=True, eet_sign=+1,
                      freq=None, redmass=None,
                      lvc_inter=None, lvc_intra=None, 
                      pbc=False):

       self.V = None                          # Hamiltonian matrix in diabatic basis (in eV)
       self.E = None                          # Adiabatic energies (in eV)
       self.U = None                          # Diabatic-to-Adiabatic transformation

       self.N = N                             # Number of monomer units
       self.M = None                          # Number of diabatic basis functions

       self.freq = freq                       # Fragment's Harmonic frequencies (cm-1)
       self.redmass = redmass                 # Fragment's Reduced masses (AMU)
       self.lvc_inter = lvc_inter             # Fragment's LVC parameters: interstate ()
       self.lvc_intra = lvc_intra             # Fragment's LVC parameters: intrastate ()

       self.m = None                          # Number of Fragment's q modes
       self.m_all = None                      # Number of all q modes

       self.trdip = None                      # Transition dipole moments in diabatic basis
       self.Trdip = None                      # Transition dipole moments in adiabatic basis

       self.v_0  = v_0                        # Ground state energy offset (in eV)
       self.e_ex = e_ex                       # Excitation energy (unperturbed; in eV)

       self.nn_v0 = nn_v0                     # NN for ground state potential
       self.nn_ex = nn_ex                     # NN for excitation energy difference potential
       self.nn_eet= nn_eet                    # NN for EET coupling

       self.eet_sign = eet_sign               # Multiply the EET coupling by eet_sign1.

       self._all_freq      = None
       self._all_fc        = None
       self._all_lvc_inter = None
       self._all_lvc_intra = None

       self.include_ground_state = include_ground_state
       self.pbc = pbc

       assert self.include_ground_state, "Now only inclusive ground state is implemented"
       assert self.pbc == False , "PBC is not implemented yet"

       self._init()

   def update(self, R, q=None):
       self.V = self.hamiltonian_el(R)
       if q is not None:
          self.V += self.hamiltonian_vib(q)
          self.V += self.hamiltonian_el_vib(R, q)
       self._diagonalize()

   def hamiltonian_el(self, R):
       "Calculate pure electronic contribution to Hamiltonian due to R"
       assert R.size == self.N-1

       dH = numpy.zeros((self.M, self.M))

       # ground state energy offset
       dH[0,0] = self.v_0 # eV

       # Ground state energy fluctuation
       if self.nn_v0 is not None:
          v0  = 0.0
          for i in range(self.N-1):
              ri = R[i]
              v0+= self.potential_v0(ri) # eV
          dH += numpy.identity(self.M) * v0

       # Excitation site energies (unperturbed)
       for i in range(self.N):
           dH[i+1,i+1] = self.e_ex # eV

       # Fluctuation of excitation site energies and EET couplings 
       for i in range(self.N-1):
           ri = R[i]
       
           dH[i+1,i+2]  = dH[i+2,i+1] \
                        = self.potential_eet(ri)  # eV
           dH[i+1,i+1] += self.potential_ex (ri)  # eV
           dH[i+2,i+2] += self.potential_ex (ri)  # eV

       return dH

   def hamiltonian_vib(self, q):
       "Calculate purely vibrational contribution to Hamiltonian due to q"
       assert q.size == self.m_all

       # Ground state harmonic potential 
       v_harm = 0.5*self._all_fc  *q*q
       dH = v_harm.sum() * numpy.identity(self.M)

       # LVC coupling contributions
       mprev = 0
       mnext = self.m
       for i in range(self.N):
           q_i = q                  [mprev:mnext]
           k_i = self._all_lvc_inter[mprev:mnext]
           l_i = self._all_lvc_intra[mprev:mnext]
           dH[i+1,i+1] += (k_i*q_i).sum()
           dH[0  ,i+1] += (l_i*q_i).sum()
           dH[i+1,0  ] += (l_i*q_i).sum()

           mprev = mnext
           mnext+= self.m
       return dH

   def hamiltonian_el_vib(self, R, q):
       "Calculate electronic-vibrational contribution to Hamiltonian due to R and q"
       assert R.size == self.N-1
       assert q.size == self.m_all

       dH = numpy.zeros((self.M, self.M))
       #TODO - this case implements scenarion where freq and lvc params depend on R (NN model of vibr solvatochromism)
       return dH

   def adpop(self, rho):
       "Calculate the adiabatic state populations based on density matrix in diabatic basis representation"
       D = self.U.T @ rho @ self.U
       pop = D.diagonal().real
       return pop

   # ---> NN potentials <--- #

   def potential_v0(self, r):
       "Returns ground state energy shift due to other fragment at r (given in a.u.)"
       return self.nn_v0.potential([[r,]])[0] * self.HartreeToElectronVolt

   def potential_ex(self, r):
       "Returns excitation energy shift due to other fragment at r (given in a.u.)"
       return self.nn_ex.potential([[r,]])[0] * self.HartreeToElectronVolt
   
   def potential_eet(self, r):
       "Returns excitation energy shift due to other fragment at r (given in a.u.)"
       return self.eet_sign * self.nn_eet.potential([[r,]])[0] * self.HartreeToElectronVolt


   # ---> protected <--- #

   def _init(self):
       "Finish the initialization"
       self.M = self.N if not self.include_ground_state else self.N+1
       self.trdip = numpy.ones(self.M)
       if self.include_ground_state: self.trdip[0] = 0.0
       if self.freq is not None:
          assert len(self.freq) == len(self.redmass) == len(self.lvc_intra) == len(self.lvc_inter)
          self.freq = numpy.array(self.freq)
          self.redmass = numpy.array(self.redmass)
          self.lvc_intra = numpy.array(self.lvc_intra)
          self.lvc_inter = numpy.array(self.lvc_inter)

          self.m = len(self.freq)
          self.m_all = self.m * self.N

          self._all_freq = numpy.zeros(self.m_all)
          self._all_fc   = numpy.zeros(self.m_all)
          self._all_lvc_intra = numpy.zeros(self.m_all)
          self._all_lvc_inter = numpy.zeros(self.m_all)

          freq        =  self.freq*self.CmRecToHartree     * self.HartreeToElectronVolt
          force_const = (self.freq*self.CmRecToHartree)**2 * self.redmass*self.AmuToElectronMass * self.HartreeToElectronVolt
          wrong_code=False
          if wrong_code:
              lvc_inter   = self.lvc_inter * self.redmass*numpy.sqrt(self.AmuToElectronMass) * self.HartreeToElectronVolt
              lvc_intra   = self.lvc_intra * self.redmass*numpy.sqrt(self.AmuToElectronMass) * self.HartreeToElectronVolt
          else:
              lvc_inter   = self.lvc_inter * numpy.sqrt(self.redmass*self.AmuToElectronMass) * self.HartreeToElectronVolt
              lvc_intra   = self.lvc_intra * numpy.sqrt(self.redmass*self.AmuToElectronMass) * self.HartreeToElectronVolt
          for i in range(self.N):
              self._all_fc       [i*self.m:(i+1)*self.m] = force_const.copy()
              self._all_freq     [i*self.m:(i+1)*self.m] = freq       .copy()
              self._all_lvc_inter[i*self.m:(i+1)*self.m] = lvc_inter.copy()
              self._all_lvc_intra[i*self.m:(i+1)*self.m] = lvc_intra.copy()
       pass

   def _diagonalize(self):
       "Diagonalize Hamiltonian"
       self.E, self.U = numpy.linalg.eigh(self.V)
       self.Trdip = self.U.T @ self.trdip

   def __repr__(self):
       "Print current state of the system"
       log = "Energy: "
       log+= self.M*"%10.3f"%tuple(self.E) + "\n"
       log+= "Tr.Dip: "
       log+= self.M*"%10.3f"%tuple(self.Trdip) + "\n"
       for i in range(self.M):
           log+= "Popul. %02dth state " % (i+1)
           log+= self.M*"%7.3f"%tuple(self.U[:,i]**2) + "\n"
       return str(log)

# parsing output utilities
def make_templ_averages_q(site,mode):
    t = r""
   #t+= r"Mode expectation values and variances :\n" 
    t+= r"q%d%d.*<q>= *(?P<average_q>-?\d+\.\d+)" % (site, mode)
    t+= r".*<dq>= *(?P<variance_q>-?\d+\.\d+)"
   #t+= r".*\n"
    return re.compile(t,          )
   #return re.compile(t, re.DOTALL)

def make_templ_averages_r(mode):
    t = r""
   #t+= r"Mode expectation values and variances :\n" 
    t+= r"r%d.*<q>= *(?P<average_r>-?\d+\.\d+)" % (mode)
    t+= r".*<dq>= *(?P<variance_r>-?\d+\.\d+)"
   #t+= r".*\n"
    return re.compile(t,          )
   #return re.compile(t, re.DOTALL)

t_time = re.compile(r" Time  = *(?P<time>\d+\.\d+) fs.*\n")
