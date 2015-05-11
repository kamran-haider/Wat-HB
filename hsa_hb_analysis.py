from __future__ import division
__doc__='''

#===============================================================================
#
#          FILE:  Main classes and functions implementing hydrogen bond calculations on a 3-D grid
                  In general this grid coincides with a GIST grid from a previously run GIST calculation.
#         USAGE:  a tester script will be provided as an example 
# 
#   DESCRIPTION:  
# 
#       OPTIONS:  ---
#  REQUIREMENTS:  Desmond, Schrodinger Python API
#          BUGS:  ---
#         NOTES:  ---
#        AUTHOR:  Kamran Haider
#   Contibutors:  
#     COPYRIGHT:  
#       COMPANY:  
#       VERSION:  1.0
#       CREATED:  
#      REVISION:  ---
#===============================================================================

'''
_version = "$Revision: 1.0 $"
# import shrodinger modules
from schrodinger import structure

from schrodinger.trajectory.desmondsimulation import create_simulation
from schrodinger.trajectory.atomselection import select_component
from schrodinger.trajectory.atomselection import FrameAslSelection as FAS
from schrodinger.trajectory.pbc_manager import PBCMeasureMananger

# import other python modules
import numpy as np
#from scipy.spatial import KDTree, cKDTree
import sys, time
#import math
degrees_per_rad = 180./np.pi

#################################################################################################################
# Main GIST class                                                                                               #
#################################################################################################################



class HBcalcs:
#*********************************************************************************************#
    # Initializer function
    def __init__(self, input_cmsname, input_trjname, clustercenter_file):
        """
        Data members
        """
        self.cmsname = input_cmsname
        self.dsim = create_simulation(input_cmsname, input_trjname)
        self._indexGenerator()
        self.hsa_data = self._initializeHSADict(clustercenter_file)
        self.box = self._initializePBC()

#*********************************************************************************************#
    # index generator function
    def _indexGenerator(self):

        frame = self.dsim.getFrame(0)
        # atom and global indices of all atoms in the system
        self.all_atom_ids = np.arange(len(self.dsim.cst.atom))+1
        self.all_atom_gids = np.arange(len(self.dsim.cst.atom)+self.dsim.cst._pseudo_total)
        # obtain oxygen atom and global indices for all water molecules
        oxygen_sel = FAS('atom.ele O')
        all_oxygen_atoms = oxygen_sel.getAtomIndices(frame)
        water_sel = select_component(self.dsim.cst, ['solvent'])
        solvent_atoms = water_sel.getAtomIndices(frame)
        solvent_oxygen_atoms = list(set(solvent_atoms).intersection(set(all_oxygen_atoms)))
        solvent_oxygen_atoms.sort()
        self.wat_oxygen_atom_ids = np.array(solvent_oxygen_atoms, dtype=np.int)
        self.wat_oxygen_atom_gids = self.wat_oxygen_atom_ids - 1
        # obtain atom indices for all water atoms
        self.wat_atom_ids = self.getWaterIndices(self.wat_oxygen_atom_ids)
        # obtain atom and global indices for all water atoms
        #wat_id_list = self.getWaterIndices(self.wat_oxygen_atom_ids)
        #self.wat_atom_ids = wat_id_list[0]
        #self.wat_atom_gids = wat_id_list[1]        
        # obtain all non-water atom and global indices
        self.non_water_atom_ids = np.setxor1d(self.all_atom_ids, self.wat_atom_ids).astype(int)
        #self.non_water_gids = np.setxor1d(self.all_atom_gids,self.wat_atom_gids)
 
#*********************************************************************************************#
    # retrieve water atom indices for selected oxygen indices 
    def getWaterIndices(self, oxygen_atids):
        # here we will get data that is required to create previous index mapper object
        # first step is to obtain solvent forcefield structure
        solvent_ffst = None
        # obtain solvent fsst by iterating over all 'types' of forcefield (i.e., solute, solvent, ion)
        for ffst in self.dsim.cst.ffsts:
            if ffst.parent_structure.property['s_ffio_ct_type'] == 'solvent':
                if solvent_ffst is not None:
                    raise Exception("does not support multiple solvent ct.")
                solvent_ffst = ffst

        # set oxygen index to none
        oxygen_index = None
        # set types of pseudo particles to 0
        npseudo_sites = 0
        # set number of solvent atom types to 0
        natom_sites = 0
       # for each forcefield site (which is any 'site' on the structure to which params are assigned)
        for i, site in enumerate(solvent_ffst.ffsite):
            # check if this site belongs to Oxygen atoms
            if site.vdwtype.upper().startswith('O'):
                # if oxygen index is already defined, raise exception otherwise set oxygen index to this site
                if oxygen_index is not None:
                    raise Exception("water molecule has more than two oxygen atoms")
                oxygen_index = i
            # check if this site belongs to pseudoparticle, if yes raise corresponding number
            if site.type.lower() == 'pseudo':
                npseudo_sites += 1
            # check if this site belongs to an atom, if yes raise corresponding number
            elif site.type.lower() == 'atom':
                natom_sites += 1
        # at the end of this loop we have checked all possible forcefield sites to get the correst index for oxygen
        # in addition we get total number of atoms and pseudopartciles on a solvent site (water in this case) 
        if oxygen_index is None:
            raise Exception("can not locate oxygen atom.")
        if natom_sites == 0:
            raise Exception("number of atoms is zero.")
        # here we totall number of atoms in solvent 
        nmols = len(solvent_ffst.parent_structure.atom)/natom_sites
        #print oxygen_index
        # this is atid for the first oxygen atom in water oxygen atom array
        wat_begin_atid = oxygen_atids[0]
        # gid in this case is atid - 1
        wat_begin_gid = wat_begin_atid - 1
        oxygen_gids = oxygen_atids - 1
        pseudo_begin_gid = wat_begin_gid + natom_sites*nmols
        id_list = []
        #return atids of atoms of selected water molecules.
        water_atids = []
        for oxygen_atid in oxygen_atids:
            for i in range(natom_sites):
                atid = oxygen_atid + i - oxygen_index
                water_atids.append(atid)
        #id_list.append(np.array(water_atids))
        #return gids of particles (including pseudo sites) of selected water molecules.
        # For now we will ignore GIDs but these are important when water model has pseudoatoms
        """
        water_gids = []
        for oxygen_gid in oxygen_gids:
            for i in range(natom_sites):
                gid = oxygen_gid + i - oxygen_index
                water_gids.append(gid)
            # pseudo atoms are placed right after real atoms
            offset = (oxygen_gid - wat_begin_gid) / natom_sites
            for i in range(npseudo_sites):
                gid = pseudo_begin_gid + offset*npseudo_sites + i
                water_gids.append(gid)
        water_gids.sort()
        id_list.append(np.array(water_gids, dtype=np.int))
        """
        self.oxygen_index = oxygen_index
        self.n_atom_sites = natom_sites
        self.n_pseudo_sites = npseudo_sites
        self.wat_begin_gid = wat_begin_gid
        self.pseudo_begin_gid = pseudo_begin_gid
        return np.array(water_atids)

#*********************************************************************************************#
    def _initializeHSADict(self, clust_center_file):
        clusters = structure.StructureReader(clust_center_file).next()
        hs_dict = {}
        cluster_centers = clusters.getXYZ()
        c_count = 0
        for h in cluster_centers: 
            hs_dict[c_count] = [tuple(h)] # create a dictionary key-value pair with voxel index as key and it's coords as
            hs_dict[c_count].append(np.zeros(14, dtype="float64"))
            hs_dict[c_count].append([]) # to store E_nbr distribution
            hs_dict[c_count].append([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [], [], [], []]) # to store hbond info
            c_count += 1
        
        return hs_dict


#*********************************************************************************************#
    def _initializePBC(self):
        # for minimum image convention
        box_vectors = self.dsim.getFrame(0).box
        if box_vectors[0] == 0.0 or box_vectors[4] == 0.0 or box_vectors[8] == 0.0:
            print "Warning: Periodic Boundary Conditions unspecified!"
        else:
            box = np.asarray([box_vectors[0], box_vectors[4], box_vectors[8]])
        return box
#*********************************************************************************************#

#*********************************************************************************************#
    def getNeighborAtoms(self, xyz, dist, point):
        """
        An efficint routine for neighbor search
        """
        # create an array of indices around a cubic grid
        dist_squared = dist * dist
        neighbors = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    neighbors.append((i,j,k))
        neighbor_array = np.array(neighbors, np.int)
        min_ = np.min(xyz, axis=0)
        cell_size = np.array([dist, dist, dist], np.float)
        cell = np.array((xyz - min_) / cell_size)#, dtype=np.int)
        # create a dictionary with keys corresponding to integer representation of transformed XYZ's
        cells = {}
        for ix, assignment in enumerate(cell):
            # convert transformed xyz coord into integer index (so coords like 1.1 or 1.9 will go to 1)
            indices =  assignment.astype(int)
            # create interger indices
            t = tuple(indices)
            # NOTE: a single index can have multiple coords associated with it
            # if this integer index is already present
            if t in cells:
                # obtain its value (which is a list, see below)
                xyz_list, trans_coords, ix_list = cells[t]
                # append new xyz to xyz list associated with this entry
                xyz_list.append(xyz[ix])
                # append new transformed xyz to transformed xyz list associated with this entry
                trans_coords.append(assignment)
                # append new array index 
                ix_list.append(ix)
            # if this integer index is encountered for the first time
            else:
                # create a dictionary key value pair,
                # key: integer index
                # value: [[list of x,y,z], [list of transformed x,y,z], [list of array indices]]
                cells[t] = ([xyz[ix]], [assignment], [ix])

        cell0 = np.array((point - min_) / cell_size, dtype=np.int)
        tuple0 = tuple(cell0)
        near = []
        for index_array in tuple0 + neighbor_array:
            t = tuple(index_array)
            if t in cells:
                xyz_list, trans_xyz_list, ix_list = cells[t]
                for (xyz, ix) in zip(xyz_list, ix_list):
                    diff = xyz - point
                    if np.dot(diff, diff) <= dist_squared and float(np.dot(diff, diff)) > 0.0:
                        #near.append(ix)
                        #print ix, np.dot(diff, diff)
                        near.append(ix)
        return near

#*********************************************************************************************#
    def _getTheta(self, frame, pos1, pos2, pos3):
        "Adapted from Schrodinger pbc_manager class"
        pos1.shape=1,3
        pos2.shape=1,3
        pos3.shape=1,3

        r21 = frame.getMinimalDifference(pos1, pos2)
        r23 = frame.getMinimalDifference(pos3, pos2)

        norm = np.sqrt(np.sum(r21**2, axis=1) *
                          np.sum(r23**2, axis=1))
        cos_theta = np.sum(r21*r23, axis=1)/norm

        # FIXME: is the isnan check sufficient?

        # handle problem when pos1 or pos3 == pos2; make theta = 0.0 in this
        # case
        if np.any(np.isnan(cos_theta)):
            cos_theta[np.isnan(cos_theta)] = 1.0

        # handle numerical roundoff issue where abs(cos_theta) may be
        # greater than 1
        if np.any(np.abs(cos_theta) > 1.0):
            cos_theta[cos_theta >  1.0] =  1.0
            cos_theta[cos_theta < -1.0] = -1.0
        theta = np.arccos(cos_theta) * degrees_per_rad
        return theta


#*********************************************************************************************#
                   
    def run_hb_analysis(self, n_frame, start_frame):
        # first step is to iterate over each frame
        for i in xrange(start_frame, start_frame + n_frame):
            print "Processing frame: ", i+1, "..."
            # get frame structure, position array
            frame = self.dsim.getFrame(i)
            #measure_manager = PBCMeasureMananger(frame)
            frame_st = self.dsim.getFrameStructure(i)
            pos = frame.position
            oxygen_pos = pos[self.wat_oxygen_atom_ids-1] # obtain coords of O-atoms
            # begin iterating over each cluster center in the cluster/HSA dictionary
            for cluster in self.hsa_data:
                #print "processin cluster: ", cluster
                nbr_indices = self.getNeighborAtoms(oxygen_pos, 1.0, self.hsa_data[cluster][0])
                cluster_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in nbr_indices]
                # begin iterating over water oxygens found in this cluster in current frame
                for wat_O in cluster_wat_oxygens:
                    self.hsa_data[cluster][1][0] += 1
                    cluster_water_all_atoms = self.getWaterIndices(np.asarray([wat_O]))
                    #print wat_O-1, pos[wat_O-1]
                    #print cluster_water_all_atoms
                    nbr_indices = self.getNeighborAtoms(oxygen_pos, 3.5, pos[wat_O-1])
                    firstshell_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in nbr_indices]
                    self.hsa_data[cluster][1][2] += len(firstshell_wat_oxygens)

                    # begin iterating over neighboring oxygens of current water oxygen
                    for nbr_O in firstshell_wat_oxygens:
                        #print nbr_O
                        nbr_wat_all_atoms = self.getWaterIndices(np.asarray([nbr_O]))
                        #print nbr_wat_all_atoms
                        # indices are offset by -1 to facilitate validation with vmd
                        #print "Processing water pair: ", wat_O-1, nbr_O-1
                        #print pos[wat_O-1], pos[nbr_O-1], pos[nbr_wat_all_atoms[1]-1]
                        #print "angle 1: ", self._getTheta(frame, pos[wat_O-1], pos[nbr_O-1], pos[nbr_wat_all_atoms[1]-1])
                        # list of all potential H-bond angles between the water-water pair
                        theta_list = [self._getTheta(frame, pos[wat_O-1], pos[nbr_O-1], pos[nbr_wat_all_atoms[1]-1]), 
                                        self._getTheta(frame, pos[wat_O-1], pos[nbr_O-1], pos[nbr_wat_all_atoms[2]-1]), 
                                        self._getTheta(frame, pos[nbr_O-1], pos[wat_O-1], pos[cluster_water_all_atoms[1]-1]), 
                                        self._getTheta(frame, pos[nbr_O-1], pos[wat_O-1], pos[cluster_water_all_atoms[2]-1])]
                        #print min(theta_list)
                        if min(theta_list) <= 35:
                            self.hsa_data[cluster][1][3] += 1

        for cluster in self.hsa_data:
            print cluster, self.hsa_data[cluster][1][0], self.hsa_data[cluster][1][0]/options.frames, self.hsa_data[cluster][1][2]/self.hsa_data[cluster][1][0], self.hsa_data[cluster][1][3]/self.hsa_data[cluster][1][0]










if (__name__ == '__main__') :

    _version = "$Revision: 0.0 $"
    
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input_cms", dest="cmsname", type="string", help="Input CMS file")
    parser.add_option("-t", "--input_trajectory", dest="trjname", type="string", help="Input trajectory directory")
    parser.add_option("-c", "--cluster_centers", dest="clusters", type="string", help="Cluster center file")
    parser.add_option("-f", "--frames", dest="frames", type="int", help="Number of frames")
    parser.add_option("-s", "--starting frame", dest="start_frame", type="int", help="Starting frame")
    parser.add_option("-o", "--output", dest="outfile", type="string", help="Output log file")
    (options, args) = parser.parse_args()
    print "Setting things up..."
    h = HBcalcs(options.cmsname, options.trjname, options.clusters)
    print "Running calculations ..."
    h.run_hb_analysis(options.frames, options.start_frame)
