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
import os, sys, time
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
        # These lists define the search space for solute water Hbond calculation
        acc_list = []
        don_list = []
        acc_don_list = []
        # To speed up calculations, we will pre-create donor atom, hydrogen pairs
        self.don_H_pair_dict = {}
        if self.non_water_atom_ids.size != 0:
            frame_st = self.dsim.getFrameStructure(0)
            for solute_at_id in self.non_water_atom_ids:
                solute_atom = frame_st.atom[solute_at_id]
                solute_atom_type = solute_atom.pdbres.strip(" ") + " " + solute_atom.pdbname.strip(" ")
                if solute_atom.element in ["O", "S"]:
                    # if O/S atom is bonded to an H, type it as donor and an acceptor
                    if "H" in [bonded_atom.element for bonded_atom in solute_atom.bonded_atoms]:
                        #print "Found donor-acceptor atom type: %i %s" % (solute_at_id, solute_atom_type)
                        acc_don_list.append(solute_at_id)
                        # create a dictionary entry for this atom, that will hold all atom, H id pairs 
                        self.don_H_pair_dict[solute_at_id] = []
                        for bonded_atom in solute_atom.bonded_atoms:
                            if bonded_atom.element == "H":
                                self.don_H_pair_dict[solute_at_id].append([solute_at_id, bonded_atom.index])

                    # if O/S atom is not bonded to an H, type it as an acceptor
                    else:
                        #print "Found acceptor atom type: %i %s" % (solute_at_id, solute_atom_type)
                        acc_list.append(solute_at_id)
                if solute_atom.element in ["N"]:
                    # if N atom is bonded to an H, type it as a donor
                    if "H" in [bonded_atom.element for bonded_atom in solute_atom.bonded_atoms]:
                        #print "Found donor atom type: %i %s" % (solute_at_id, solute_atom_type)
                        don_list.append(solute_at_id)
                        # create a dictionary entry for this atom, that will hold all atom, H id pairs 
                        self.don_H_pair_dict[solute_at_id] = []
                        for bonded_atom in solute_atom.bonded_atoms:
                            if bonded_atom.element == "H":
                                self.don_H_pair_dict[solute_at_id].append([solute_at_id, bonded_atom.index])


                    # if N atom is not bonded to an H, type it as an acceptor
                    else:
                        #print "Found acceptor atom type: %i %s" % (solute_at_id, solute_atom_type)
                        acc_list.append(solute_at_id)
        #print don_list
        #print acc_list
        #print acc_don_list
        self.solute_acc_ids = np.array(acc_list, dtype=np.int)
        self.solute_acc_don_ids = np.array(acc_don_list, dtype=np.int)
        self.solute_don_ids = np.array(don_list, dtype=np.int)
        #for at in self.don_H_pair_dict:
            #print at, self.don_H_pair_dict[at]

 
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
        data_fields = 15
        self.data_titles = ["wat", "occ", "gO", "nbrs", 
                        "HBww", "HBsw", "HBtot", "Acc_ww", "Don_ww", "Acc_sw", "Don_sw", 
                        "percentHBww", "soluteHBnbrs", "percentHBsw", "enclosure", 
                        "HBangleww", "HBanglesw"]

        for h in cluster_centers: 
            hs_dict[c_count] = [tuple(h)] # create a dictionary key-value pair with voxel index as key and it's coords as
            hs_dict[c_count].append(np.zeros(data_fields, dtype="float64"))
            hs_dict[c_count].append([]) # to store E_nbr distribution
            for i in range(data_fields+2): hs_dict[c_count][2].append([]) 
            hs_dict[c_count].append([]) # to store hbond info (protein acceptors)
            hs_dict[c_count].append([]) # to store hbond info (protein donors)
            hs_dict[c_count].append(np.zeros(data_fields+2, dtype="float64")) # to store error info on each timeseries
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
        "Adapted from Schrodinger pbc_manager class."
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
            acc_pos = pos[self.solute_acc_ids-1] # obtain coords of O-atoms
            acc_don_pos = pos[self.solute_acc_don_ids-1] # obtain coords of O-atoms
            don_pos = pos[self.solute_don_ids-1] # obtain coords of O-atoms
            d_clust = _DistanceCell(oxygen_pos, 1.0)
            d_nbrs = _DistanceCell(oxygen_pos, 3.5)
            if self.non_water_atom_ids.size != 0:
                if self.solute_acc_ids.size != 0:
                    d_acc = _DistanceCell(acc_pos, 3.5)
                if self.solute_don_ids.size != 0:
                    d_don = _DistanceCell(don_pos, 3.5)
                if self.solute_acc_don_ids.size != 0:
                    d_acc_don = _DistanceCell(acc_don_pos, 3.5)

            # begin iterating over each cluster center in the cluster/HSA dictionary
            for cluster in self.hsa_data:
                #print "processin cluster: ", cluster
                nbr_indices = d_clust.query_nbrs(self.hsa_data[cluster][0])
                cluster_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in nbr_indices]
                
                # begin iterating over water oxygens found in this cluster in current frame
                for wat_O in cluster_wat_oxygens:
                    wat_nbrs = 0
                    solute_Hbond_partners = 0
                    hbww = 0
                    hbsw = 0
                    hbtot = 0
                    percent_hbww = 0
                    percent_hbsw = 0
                    enclosure = 0
                    self.hsa_data[cluster][1][0] += 1 # raise water population by 1
                    cluster_water_all_atoms = self.getWaterIndices(np.asarray([wat_O]))
                    #print wat_O-1, pos[wat_O-1]
                    #print cluster_water_all_atoms
                    #*********************************************************************************************************************************#
                    # The following loop is for Hbondww calculations
                    nbr_indices = d_nbrs.query_nbrs(pos[wat_O-1])
                    firstshell_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in nbr_indices]
                    wat_nbrs += len(firstshell_wat_oxygens)
                    self.hsa_data[cluster][1][3] += wat_nbrs # add  to cumulative sum
                    self.hsa_data[cluster][2][3].append(wat_nbrs) # add nbrs to nbr timeseries
                    # Enclosure calculations
                    enclosure += 1 - (wat_nbrs/5.25)
                    if enclosure < 0.0:
                        enclosure = 0.0
                    self.hsa_data[cluster][1][14] += enclosure # add value to cumulative sum
                    self.hsa_data[cluster][2][14].append(enclosure) # append to timeseries

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
                        # theta angle = Acceptor-Donor-Hydrogen
                        theta_list = [self._getTheta(frame, pos[wat_O-1], pos[nbr_O-1], pos[nbr_wat_all_atoms[1]-1]), 
                                        self._getTheta(frame, pos[wat_O-1], pos[nbr_O-1], pos[nbr_wat_all_atoms[2]-1]), 
                                        self._getTheta(frame, pos[nbr_O-1], pos[wat_O-1], pos[cluster_water_all_atoms[1]-1]), 
                                        self._getTheta(frame, pos[nbr_O-1], pos[wat_O-1], pos[cluster_water_all_atoms[2]-1])]
                        hbangle = min(theta_list) # min angle is a potential Hbond
                        #print hbangle
                        self.hsa_data[cluster][2][15].append(hbangle[0]) # add to HBangleww timeseries (regardless of whether it is an Hbond or not)
                        if hbangle <= 30: # if Hbond is made
                            #print hbangle
                            self.hsa_data[cluster][1][4] += 1 # add to cumulative sum of HBww
                            hbww += 1
                            # furthermore check if cluster water is acting as an acceptoir
                            if theta_list.index(hbangle) in [0,1]:
                                self.hsa_data[cluster][1][7] += 1 # add to cumulative of summ of HBwwacc
                            # or as a donor
                            else:
                                self.hsa_data[cluster][1][8] += 1 # add to cumulative of summ of HBwwdon
                    
                    self.hsa_data[cluster][2][4].append(hbww) # append to HBww timeseries
                    # percentHBww calculations
                    if wat_nbrs != 0:
                        percent_hbww += (hbww/wat_nbrs)*100
                    self.hsa_data[cluster][2][11].append(percent_hbww) # append to percentHBww timeseries

                    #*********************************************************************************************************************************#
                    # The following loop is for Hbondsw calculations
                    # Divided in three steps, first solute acceptors, solute acceptor-donors and then solute donors
                    # retrieve neighbor protein acceptor atoms
                    
                    if self.non_water_atom_ids.size != 0: 
                        if self.solute_acc_ids.size != 0:
                            solute_acc_nbr_indices = d_acc.query_nbrs(pos[wat_O-1])
                            solute_acceptors = [self.solute_acc_ids[nbr_index] for nbr_index in solute_acc_nbr_indices]
                            solute_Hbond_partners += len(solute_acceptors)


                            # for each acceptor atom, find potential Hbonding angles with water
                            for solute_acceptor in solute_acceptors:
                                theta_list = [self._getTheta(frame, pos[solute_acceptor-1], pos[wat_O-1], pos[cluster_water_all_atoms[1]-1]), 
                                                self._getTheta(frame, pos[solute_acceptor-1], pos[wat_O-1], pos[cluster_water_all_atoms[2]-1])]
                                hbangle = min(theta_list) # min angle is a potential Hbond
                                self.hsa_data[cluster][2][16].append(hbangle[0]) # add to HBanglesw timeseries (regardless of whether it is an Hbond or not)
                                if hbangle <= 30: # if Hbond is made
                                    hbsw += 1
                                    #print "Hbond made between: ", solute_acceptor-1, wat_O-1, hbangle
                                    self.hsa_data[cluster][1][5] += 1 #add to cumulative sum of HBsw
                                    self.hsa_data[cluster][1][10] += 1 # add to cumulative of summ of HBswdon
                                    self.hsa_data[cluster][3].append(solute_acceptor)
                            # retrieve neighbor protein donor atoms
                        if self.solute_don_ids.size != 0:
                            solute_don_nbr_indices = d_don.query_nbrs(pos[wat_O-1])
                            solute_donors = [self.solute_don_ids[nbr_index] for nbr_index in solute_don_nbr_indices]
                            solute_Hbond_partners += len(solute_donors)
                            # for each donor atom, find potential Hbonding angles with water
                            for solute_donor in solute_donors:
                                theta_list = []
                                for solute_don_H_pair in self.don_H_pair_dict[solute_donor]:
                                    #print solute_don_H_pair
                                    theta_list.append(self._getTheta(frame, pos[wat_O-1], pos[solute_don_H_pair[0]-1], pos[solute_don_H_pair[1]-1]))
                                hbangle = min(theta_list) # min angle is a potential Hbond
                                self.hsa_data[cluster][2][16].append(hbangle[0]) # add to HBanglesw timeseries (regardless of whether it is an Hbond or not)
                                if hbangle <= 30: # if Hbond is made
                                    hbsw += 1
                                    #print "Hbond made between: ", solute_donor-1, wat_O-1, hbangle
                                    self.hsa_data[cluster][1][5] += 1 # add to cumulative sum of HBsw
                                    self.hsa_data[cluster][1][9] += 1 # add to cumulative of summ of HBswacc
                                    self.hsa_data[cluster][4].append(solute_donor)

                        if self.solute_acc_don_ids.size != 0:

                            # retrieve neighbor protein atoms that can act as both donors and acceptors
                            solute_acc_don_nbr_indices = d_acc_don.query_nbrs(pos[wat_O-1])
                            solute_acceptors_donors = [self.solute_acc_don_ids[nbr_index] for nbr_index in solute_acc_don_nbr_indices]
                            solute_Hbond_partners += len(solute_acceptors_donors)
                            for solute_acc_don in solute_acceptors_donors:
                                for solute_acc_don_H_pair in self.don_H_pair_dict[solute_acc_don]:
                                    #print solute_acc_don_H_pair
                                    theta_list = [self._getTheta(frame, pos[wat_O-1], pos[solute_acc_don_H_pair[0]-1], pos[solute_acc_don_H_pair[1]-1]), 
                                                    self._getTheta(frame, pos[solute_acc_don_H_pair[0]-1], pos[wat_O-1], pos[cluster_water_all_atoms[1]-1]), 
                                                    self._getTheta(frame, pos[solute_acc_don_H_pair[0]-1], pos[wat_O-1], pos[cluster_water_all_atoms[2]-1])]
                                    hbangle = min(theta_list) # min angle is a potential Hbond
                                    self.hsa_data[cluster][2][16].append(hbangle[0]) # add to HBanglesw timeseries (regardless of whether it is an Hbond or not)
                                    if hbangle <= 30: # if Hbond is made
                                        hbsw += 1
                                        #print "Hbond made between: ", solute_acc_don-1, wat_O-1, hbangle
                                        self.hsa_data[cluster][1][5] += 1 #add to cumulative sum of HBsw
                                        # furthermore check if cluster water is acting as an acceptoir
                                        if theta_list.index(hbangle) == 0:
                                            self.hsa_data[cluster][1][9] += 1 # add to cumulative of summ of HBswacc
                                            self.hsa_data[cluster][4].append(solute_acc_don)
                                        # or as a donor
                                        else:
                                            self.hsa_data[cluster][1][10] += 1 # add to cumulative of summ of HBswdon
                                        self.hsa_data[cluster][3].append(solute_acc_don)
                        
                        # add to the cumulative of solute H-bonding neighbors
                        self.hsa_data[cluster][1][12] += solute_Hbond_partners
                        self.hsa_data[cluster][2][12].append(solute_Hbond_partners) # append to solute HB partners timeseries

                        self.hsa_data[cluster][2][5].append(hbsw) # append to solute HBonds timeseries

                        self.hsa_data[cluster][1][6] += hbww+hbsw # add to cumulative sum of total Hbonds
                        self.hsa_data[cluster][2][6].append(hbww+hbsw) # append to total Hbond timeseries

                        if solute_Hbond_partners != 0:
                            percent_hbsw += (hbsw/solute_Hbond_partners)*100

                        self.hsa_data[cluster][1][13] += percent_hbsw # add to cumulative sum of percentHBsw
                        self.hsa_data[cluster][2][13].append(percent_hbsw) # append to percentHBsw timeseries
                    
#*********************************************************************************************#
                   

#*********************************************************************************************#

    def normalizeClusterQuantities(self, n_frame):
        rho_bulk = 0.0329 #molecules/A^3 # 0.0329
        sphere_vol = (4/3)*np.pi*1.0
        bulkwaterpersite = rho_bulk*n_frame*sphere_vol
        for cluster in self.hsa_data:
            if self.hsa_data[cluster][1][0] != 0:
                #print cluster
                # occupancy of the cluster
                self.hsa_data[cluster][1][1] = self.hsa_data[cluster][1][0]/n_frame
                # gO of the cluster
                self.hsa_data[cluster][1][2] = self.hsa_data[cluster][1][0]/(bulkwaterpersite)
                # normalized number of neighbors
                self.hsa_data[cluster][1][3] /= self.hsa_data[cluster][1][0]
                # normalized number of HBww
                self.hsa_data[cluster][1][4] /= self.hsa_data[cluster][1][0]
                if self.non_water_atom_ids.size != 0:
                    # normalized number of HBsw
                    self.hsa_data[cluster][1][5] /= self.hsa_data[cluster][1][0]
                    # Calculate %Accsw
                    self.hsa_data[cluster][1][9] = (self.hsa_data[cluster][1][9]/self.hsa_data[cluster][1][0])*100
                    # Calculate %Donsw
                    self.hsa_data[cluster][1][10] = (self.hsa_data[cluster][1][10]/self.hsa_data[cluster][1][0])*100
                    # Calculate %HBsw, first normalize protein h-bonding nbrs
                    if self.hsa_data[cluster][1][12] != 0:
                        self.hsa_data[cluster][1][12] /= self.hsa_data[cluster][1][0]
                        self.hsa_data[cluster][1][13] = (self.hsa_data[cluster][1][5]/self.hsa_data[cluster][1][12])*100

                
                # normalized number of HBtot
                self.hsa_data[cluster][1][6] = self.hsa_data[cluster][1][4] + self.hsa_data[cluster][1][5]
                # Calculate %Accww
                self.hsa_data[cluster][1][7] = (self.hsa_data[cluster][1][7]/self.hsa_data[cluster][1][0])*100
                # Calculate %Donww
                self.hsa_data[cluster][1][8] = (self.hsa_data[cluster][1][8]/self.hsa_data[cluster][1][0])*100
                # Calculate %HBww
                self.hsa_data[cluster][1][11] = (self.hsa_data[cluster][1][4]/self.hsa_data[cluster][1][3])*100
                # normalized enclosure
                self.hsa_data[cluster][1][14] /= self.hsa_data[cluster][1][0]
                
                #print self.hsa_data[cluster][1][0], self.hsa_data[cluster][1][1]
                # Obtain set of protein residues involved in solute-water H-bonding
                hs_acc_residues = []
                hs_don_residues = []
               
                if self.hsa_data[cluster][1][5] != 0:
                    cluster_acceptors = np.unique(np.asarray(self.hsa_data[cluster][3]))
                    cluster_donors = np.unique(np.asarray(self.hsa_data[cluster][4]))
                    for sw_acc in cluster_acceptors:
                        acc_residue = self.dsim.cst.atom[sw_acc].getResidue().pdbres.strip() + str(self.dsim.cst.atom[sw_acc].getResidue().resnum)
                        hs_acc_residues.append(acc_residue)
                    for sw_don in cluster_donors:
                        don_residue = self.dsim.cst.atom[sw_don].getResidue().pdbres.strip() + str(self.dsim.cst.atom[sw_don].getResidue().resnum)
                        hs_don_residues.append(don_residue)
                self.hsa_data[cluster][3] = hs_acc_residues
                self.hsa_data[cluster][4] = hs_don_residues
                #print hs_acc_residues
                #print hs_don_residues 



#*********************************************************************************************#

    def writeHBsummary(self, prefix):
        f = open(prefix+"_hsa_hb_summary.txt", "w")
        header_2 = "index x y z wat occ gO nbrs HBww HBsw HBtot Enclosure percentHBww soluteHBnbrs percentHBsw Acc_ww Don_ww Acc_sw Don_sw Solute-Don Solute-Acc\n"
        f.write(header_2)
        for cluster in self.hsa_data:
            d = self.hsa_data[cluster]
            # create don and acceptor strings
            acc_string = " "
            don_string = " "
            for acc in d[3]:
                acc = acc + ","
                acc_string += acc
            for don in d[4]:
                don = don + ","
                don_string += don
            #print don_string, acc_string
            l = "%d %.2f %.2f %.2f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s %s\n" % \
                ( cluster, d[0][0], d[0][1], d[0][2], \
                d[1][0], d[1][1], d[1][2], \
                d[1][3], d[1][4], d[1][5], d[1][6], d[1][14],\
                d[1][11], d[1][12], d[1][13], \
                d[1][7], d[1][8], d[1][9], d[1][10], \
                don_string, acc_string)
            f.write(l)
        f.close()

        e = open(prefix+"_hsa_hb_stats.txt", "w")
        header_3 = "index nbrs HBww HBsw HBtot Enclosure percentHBww soluteHBnbrs percentHBsw Acc_ww Don_ww Acc_sw Don_sw\n"
        e.write(header_3)
        for cluster in self.hsa_data:
            d = self.hsa_data[cluster]
            l = "%d %f %f %f %f %f %f %f %f %f %f %f %f\n" % \
                ( cluster, d[5][3], d[5][4], d[5][5], d[5][6], d[5][14],\
                d[5][11], d[5][12], d[5][13], \
                d[5][7], d[5][8], d[5][9], d[5][10])
            #print l
            e.write(l)
        e.close()


#*********************************************************************************************#

    def writeTimeSeries(self, prefix):
        cwd = os.getcwd()
        # create directory to store detailed data for individual columns in HSA
        directory = cwd + "/" + prefix+"_cluster_hb_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)
        # for each cluster, go through time series data
        # for each cluster, go through time series data
        for cluster in self.hsa_data:
            cluster_index = "%03d_" % cluster
            #print cluster_index#, self.hsa_data[cluster][2]
            for index, data_field in enumerate(self.hsa_data[cluster][2]):
                # only write timeseries data that was stored during calculation
                if len(data_field) != 0:
                    # create appropriate file name
                    data_file = cluster_index + prefix + "_" + self.data_titles[index] 
                    #print index, self.data_titles[index]
                    f =  open(data_file, "w")
                    # write each value from the timeseries into the file
                    for value in data_field:
                    #    print value
                        f.write(str(value)+"\n")
                    f.close()
                    self.hsa_data[cluster][5][index] += np.std(np.asarray(data_field))
        os.chdir("../")


#################################################################################################################
# Class and methods for 'efficient' neighbor search                                                             #
#################################################################################################################

class _DistanceCell:
    def __init__(self, xyz, dist):
        """
        Class for fast queries of coordinates that are within distance <dist>
        of specified coordinate. This class must first be initialized from an
        array of all available coordinates, and a distance threshold. The
        query() method can then be used to get a list of points that are within
        the threshold distance from the specified point.
        """
        # create an array of indices around a cubic grid
        self.neighbors = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    self.neighbors.append((i,j,k))
        self.neighbor_array = np.array(self.neighbors, np.int)

        self.min_ = np.min(xyz, axis=0)
        self.cell_size = np.array([dist, dist, dist], np.float)
        cell = np.array((xyz - self.min_) / self.cell_size)#, dtype=np.int)
        # create a dictionary with keys corresponding to integer representation of transformed XYZ's
        self.cells = {}
        for ix, assignment in enumerate(cell):
            # convert transformed xyz coord into integer index (so coords like 1.1 or 1.9 will go to 1)
            indices =  assignment.astype(int)
            # create interger indices
            t = tuple(indices)
            # NOTE: a single index can have multiple coords associated with it
            # if this integer index is already present
            if t in self.cells:
                # obtain its value (which is a list, see below)
                xyz_list, trans_coords, ix_list = self.cells[t]
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
                self.cells[t] = ([xyz[ix]], [assignment], [ix])

        self.dist_squared = dist * dist


    def query_nbrs(self, point):
        """
        Given a coordinate point, return all point indexes (0-indexed) that
        are within the threshold distance from it.
        """
        cell0 = np.array((point - self.min_) / self.cell_size, 
                                     dtype=np.int)
        tuple0 = tuple(cell0)
        near = []
        for index_array in tuple0 + self.neighbor_array:
            t = tuple(index_array)
            if t in self.cells:
                xyz_list, trans_xyz_list, ix_list = self.cells[t]
                for (xyz, ix) in zip(xyz_list, ix_list):
                    diff = xyz - point
                    if np.dot(diff, diff) <= self.dist_squared and float(np.dot(diff, diff)) > 0.0:
                        #near.append(ix)
                        #print ix, np.dot(diff, diff)
                        near.append(ix)
        return near


#*********************************************************************************************#




if (__name__ == '__main__') :

    _version = "$Revision: 0.0 $"
    
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input_cms", dest="cmsname", type="string", help="Input CMS file")
    parser.add_option("-t", "--input_trajectory", dest="trjname", type="string", help="Input trajectory directory")
    parser.add_option("-c", "--cluster_centers", dest="clusters", type="string", help="Cluster center file")
    parser.add_option("-f", "--frames", dest="frames", type="int", help="Number of frames")
    parser.add_option("-s", "--starting frame", dest="start_frame", type="int", help="Starting frame")
    parser.add_option("-o", "--output_name", dest="prefix", type="string", help="Output log file")
    (options, args) = parser.parse_args()
    print "Setting things up..."
    h = HBcalcs(options.cmsname, options.trjname, options.clusters)
    print "Running calculations ..."
    t = time.time()
    h.run_hb_analysis(options.frames, options.start_frame)
    h.normalizeClusterQuantities(options.frames)
    print "Done! took %8.3f seconds." % (time.time() - t)
    print "Writing timeseries data ..."
    h.writeTimeSeries(options.prefix)
    print "Writing summary..."
    h.writeHBsummary(options.prefix)
    