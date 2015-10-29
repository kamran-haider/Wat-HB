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
#from schrodinger.trajectory.pbc_manager import PBCMeasureMananger

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
    def __init__(self, input_cmsname, input_trjname, ligand_file):
        """
        Data members
        """
        self.cmsname = input_cmsname
        self.dsim = create_simulation(input_cmsname, input_trjname)
        self._indexGenerator(ligand_file)

        #self.hsa_data = self._initializeHSADict(clustercenter_file)
        self.box = self._initializePBC()
#*********************************************************************************************#
    # index generator function
    def _indexGenerator(self, ligand_file):
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
        # here add commands to restrict non-water atom ids to within 5.0A of ligand
        solute_indices_near_lig = []
        ligand = structure.StructureReader(ligand_file).next()
        lig_atom_coords = ligand.getXYZ()
        solute_pos = frame.position[self.non_water_atom_ids-1]
        d_solute_nbrs = _DistanceCell(solute_pos, 5)
        for coord in lig_atom_coords:
            solute_nbr_indices = d_solute_nbrs.query_nbrs(coord)
            for solute_nbr_index in solute_nbr_indices:
                if solute_nbr_index not in solute_indices_near_lig:
                    solute_indices_near_lig.append(solute_nbr_index)
        #print self.non_water_atom_ids
        self.non_water_atom_ids_near_lig = [self.non_water_atom_ids[nbr_index] for nbr_index in solute_indices_near_lig]


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
        self.prot_hbond_data = {}
        for hb_group in np.concatenate((self.solute_acc_ids, self.solute_acc_don_ids, self.solute_don_ids)):
            hb_group_name = self.dsim.cst.atom[hb_group].getResidue().pdbres.strip() + str(self.dsim.cst.atom[hb_group].getResidue().resnum) + " " + self.dsim.cst.atom[hb_group].pdbname.strip()
            self.prot_hbond_data[hb_group] = [hb_group_name, np.zeros(6, dtype="float64")]
            #print self.prot_hbond_data[hb_group]

 
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
        return theta[0]

#*********************************************************************************************#
                   
    def run_hb_analysis_sw(self, n_frame, start_frame):
        for i in xrange(start_frame, start_frame + n_frame):
            print "Processing frame: ", i+1, "..."
            frame_hb = 0 # 
            # get frame structure, position array
            frame = self.dsim.getFrame(i)
            frame_st = self.dsim.getFrameStructure(i)
            pos = frame.position
            oxygen_pos = pos[self.wat_oxygen_atom_ids-1] # obtain coords of O-atoms
            # create distance query object that searches for water neighbors for a given solute atom
            d_wat_nbrs = _DistanceCell(oxygen_pos, 3.5)
            prot_acc_pos = pos[np.concatenate((self.solute_acc_ids, self.solute_acc_don_ids)) - 1] # obtain coords of protein acceptor
            prot_don_pos = pos[np.concatenate((self.solute_don_ids, self.solute_acc_don_ids)) - 1] # obtain coords of protein acceptor
            # create distance query object that searches for protein neighbors for a given solute atom
            if self.solute_acc_ids.size != 0:
                d_prot_acc = _DistanceCell(prot_acc_pos, 3.5)
            if self.solute_don_ids.size != 0:
                d_prot_don = _DistanceCell(prot_don_pos, 3.5)
            #*****************************************************************#
            # Calculate protein H-bonds with other protein groups 
            #*****************************************************************#
            # begin iterating over solute acceptor atoms
            for solute_acceptor in np.concatenate((self.solute_acc_ids, self.solute_acc_don_ids)):
                # this condition restricts calculation to groups near ligand
                if solute_acceptor in self.non_water_atom_ids_near_lig:
                    # obtain water neighbors (oxygens) 
                    wat_nbr_indices = d_wat_nbrs.query_nbrs(pos[solute_acceptor -1])
                    nbr_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in wat_nbr_indices]
                    self.prot_hbond_data[solute_acceptor][1][0] += len(nbr_wat_oxygens)
                    if len(nbr_wat_oxygens) > 0:
                        self.prot_hbond_data[solute_acceptor][1][1] += 1
                    # iterate over each neighbor water oxygen for this acceptor
                    for wat_O in nbr_wat_oxygens:
                        nbr_water_all_atoms = self.getWaterIndices(np.asarray([wat_O])) # obtain rest of water atoms
                        # create all possible angles between current acceptor and current water
                        theta_list = [self._getTheta(frame, pos[solute_acceptor-1], pos[wat_O-1], pos[nbr_water_all_atoms[1]-1]), 
                                        self._getTheta(frame, pos[solute_acceptor-1], pos[wat_O-1], pos[nbr_water_all_atoms[2]-1])]
                        hbangle = min(theta_list) # min angle is a potential Hbond
                        if hbangle <= 30: # if Hbond is made
                            self.prot_hbond_data[solute_acceptor][1][2] += 1
            # begin iterating over solute donor atoms
            for solute_donor in np.concatenate((self.solute_don_ids, self.solute_acc_don_ids)):
                if solute_donor in self.non_water_atom_ids_near_lig:
                #for solute_donor in [2676, 2678, 548, 551, 2122]:
                #for solute_donor in [548]:
                    # obtain water neighbors for this donor
                    wat_nbr_indices = d_wat_nbrs.query_nbrs(pos[solute_donor -1])
                    nbr_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in wat_nbr_indices]
                    self.prot_hbond_data[solute_donor][1][0] += len(nbr_wat_oxygens)
                    if len(nbr_wat_oxygens) > 0:
                        self.prot_hbond_data[solute_donor][1][1] += 1
                    # iterate over each neighbor water oxygen for this donor
                    for wat_O in nbr_wat_oxygens:
                        for solute_don_H_pair in self.don_H_pair_dict[solute_donor]:
                            #print solute_don_H_pair
                            theta = self._getTheta(frame, pos[wat_O-1], pos[solute_don_H_pair[0]-1], pos[solute_don_H_pair[1]-1])
                            #theta_2 = self._getTheta(frame, pos[solute_don_H_pair[0]-1], pos[solute_don_H_pair[1]-1], pos[wat_O-1])
                            if theta <= 20:
                                #print "Hbond made between: ", solute_donor-1, wat_O-1, hbangle
                                self.prot_hbond_data[solute_donor][1][3] += 1
            #*****************************************************************#
            # Calculate protein H-bonds with other protein groups 
            #*****************************************************************#
            # begin iterating over solute acceptor atoms
            for solute_acceptor in np.concatenate((self.solute_acc_ids, self.solute_acc_don_ids)):
                if solute_acceptor in self.non_water_atom_ids_near_lig:
                    # obtain neighboring protein donors 
                    prot_don_nbr_indices = d_prot_don.query_nbrs(pos[solute_acceptor -1])
                    prot_don_nbrs = [np.concatenate((self.solute_don_ids, self.solute_acc_don_ids))[nbr_index] for nbr_index in prot_don_nbr_indices]
                    #print prot_don_nbrs
                    # iterate over each neighboring protein donor for this acceptor
                    for don in prot_don_nbrs:
                        # for each neighboring protein donor, iterate over each D-H group (e.g., Lys NH_3{+} has three N-H groups)
                        for solute_don_H_pair in self.don_H_pair_dict[don]:
                            # obtain acceptoor-donor-hydrogen angle
                            #print solute_don_H_pair
                            theta = self._getTheta(frame, pos[solute_acceptor-1], pos[solute_don_H_pair[0]-1], pos[solute_don_H_pair[1]-1])
                            #theta_2 = self._getTheta(frame, pos[solute_don_H_pair[0]-1], pos[solute_don_H_pair[1]-1], pos[wat_O-1])
                            if theta <= 20:
                                #print "Hbond made between: ", solute_donor-1, wat_O-1, hbangle
                                self.prot_hbond_data[solute_acceptor][1][4] += 1

            for solute_donor in np.concatenate((self.solute_don_ids, self.solute_acc_don_ids)):
            #for solute_donor in [3371]:
                if solute_donor in self.non_water_atom_ids_near_lig:
                    # obtain neighboring protein acceptors
                    prot_acc_nbr_indices = d_prot_acc.query_nbrs(pos[solute_donor -1])
                    prot_acc_nbrs = [np.concatenate((self.solute_acc_ids, self.solute_acc_don_ids))[nbr_index] for nbr_index in prot_acc_nbr_indices]
                    # iterate over each D-H group of this donor
                    #print np.concatenate((self.solute_acc_ids, self.solute_acc_don_ids))
                    for solute_don_H_pair in self.don_H_pair_dict[solute_donor]:
                        # check current D-H group with each neighboring acceptor
                        for acc in prot_acc_nbrs:
                            
                            # obtain acceptoor-donor-hydrogen angle
                            theta = self._getTheta(frame, pos[solute_don_H_pair[1]-1], pos[solute_don_H_pair[0]-1], pos[acc-1])
                            #print solute_don_H_pair[1]-1, solute_don_H_pair[0]-1, acc-1, theta
                            #theta_2 = self._getTheta(frame, pos[solute_don_H_pair[0]-1], pos[solute_don_H_pair[1]-1], pos[wat_O-1])
                            if theta <= 20:
                                #print "Hbond made between: ", solute_donor-1, acc-1, theta
                                self.prot_hbond_data[solute_donor][1][5] += 1

#*********************************************************************************************#

    def normalizeClusterQuantities(self, n_frame):
        for hb_group in self.prot_hbond_data:
            if hb_group in self.non_water_atom_ids_near_lig:
                self.prot_hbond_data[hb_group][1][0] /= n_frame
                self.prot_hbond_data[hb_group][1][2] /= n_frame
                self.prot_hbond_data[hb_group][1][3] /= n_frame
                self.prot_hbond_data[hb_group][1][4] /= n_frame
                self.prot_hbond_data[hb_group][1][5] /= n_frame


#*********************************************************************************************#

    def writeHBsummary(self, prefix):
        f = open(prefix+"_prot_hb_summary.txt", "w")
        header = "atom_index residue_name atom_name wat_nbrs hb_acc_sw hb_don_sw hb_acc_ss hb_don_ss\n"
        f.write(header)
        for hb_group in self.prot_hbond_data:
            if hb_group in self.non_water_atom_ids_near_lig:
                d = self.prot_hbond_data[hb_group][1]
                l = "%i %s %f %f %f %f %f\n" % (hb_group, self.prot_hbond_data[hb_group][0], d[0], d[2], d[3], d[4], d[5])
                f.write(l)
        f.close()


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
    parser.add_option("-l", "--ligand", dest="ligand_file", type="string", help="Ligand file")
    parser.add_option("-f", "--frames", dest="frames", type="int", help="Number of frames")
    parser.add_option("-s", "--starting frame", dest="start_frame", type="int", help="Starting frame")
    parser.add_option("-o", "--output_name", dest="prefix", type="string", help="Output log file")
    (options, args) = parser.parse_args()
    print "Setting things up..."
    h = HBcalcs(options.cmsname, options.trjname, options.ligand_file)
    print "Running calculations ..."
    t = time.time()
    h.run_hb_analysis_sw(options.frames, options.start_frame)
    h.normalizeClusterQuantities(options.frames)
    print "Done! took %8.3f seconds." % (time.time() - t)
    print "Writing summary..."
    h.writeHBsummary(options.prefix)
    